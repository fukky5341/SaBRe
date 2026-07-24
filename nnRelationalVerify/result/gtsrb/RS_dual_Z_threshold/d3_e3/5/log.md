## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 24.3214673868


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7968674, 33.7968674)
1: (-1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7998238, 23.7998199)
2: (1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.5222816, 20.5222836)
3: (-12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6608658, 23.6608658)
4: (-1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4174805, 24.4174805)
5: (0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7726517, 20.7726498)
6: (-74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2831078, 29.2831116)
7: (-5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.1085358, 20.1085339)
8: (-8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0708466, 28.0708466)
9: (-26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6756325, 30.6756287)
10: (-35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2160492, 37.2160492)
11: (-25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6155586, 27.6155548)
12: (-42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.2355194, 26.2355232)
13: (-38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4478302, 43.4478264)
14: (1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.6230469, 39.6230469)
15: (-17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9882278, 31.9882278)
16: (-38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3512268, 34.3512268)
17: (9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1537323, 36.1537361)
18: (-8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4335632, 41.4335632)
19: (-18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.7044067, 30.7044106)
20: (-16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9442902, 31.9442902)
21: (-16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4711838, 33.4711838)
22: (-2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6360550, 27.6360588)
23: (-14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6951447, 28.6951485)
24: (-8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5435104, 33.5435104)
25: (1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2883148, 34.2883186)
26: (-16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.6963196, 36.6963196)
27: (-14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7976761, 26.7976723)
28: (-5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2393532, 29.2393570)
29: (-6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1659622, 26.1659584)
30: (-12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4929810, 30.4929810)
31: (-17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3731689, 39.3731766)
32: (-70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.8226318, 30.8226357)
33: (-110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.4129715, 45.4129791)
34: (-88.7714386, -42.1550407, -88.7714386, -42.1550407, -29.0887413, 29.0887413)
35: (-66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.8786850, 30.8786812)
36: (-60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5683708, 29.5683746)
37: (-121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.2391968, 46.2391968)
38: (-70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7688828, 39.7688904)
39: (-111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.5427704, 41.5427780)
40: (-114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7053299, 35.7053299)
41: (-87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5434837, 29.5434875)
42: (-72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1331406, 30.1331406)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.01 + 57.64 = 60.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 17, lower bound: -24.3458132, upper bound: 24.3458132

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1693

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3015274, upper bound: 24.3434280
time: 66.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3434280, upper bound: 24.3015274
time: 45.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 112.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 112.82
Output dim: 17, lower bound: -24.3015274, upper bound: 24.3434280
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 112.82
Output dim: 17, lower bound: -24.3434280, upper bound: 24.3015274

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7926369, 33.7893105
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7948494, 23.7913094
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.5180397, 20.5146999
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6590042, 23.6585693
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4170303, 24.4168053
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7703438, 20.7691841
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2615547, 29.2729721
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.1080818, 20.1078434
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0580597, 28.0488472
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6716537, 30.6722832
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2142029, 37.2146149
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6163330, 27.6149406
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.2110176, 26.2208824
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4186325, 43.4311295
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.6167374, 39.6108704
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9819717, 31.9802437
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3433990, 34.3381042
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1461182, 36.1493835
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4245911, 41.4176254
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.7058487, 30.7036629
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9458313, 31.9439392
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4765549, 33.4699326
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6341324, 27.6348076
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7013550, 28.6936913
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5458527, 33.5421753
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2947540, 34.2866936
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.6986389, 36.6902695
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7892075, 26.7829151
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2423096, 29.2382393
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1653061, 26.1654739
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4841843, 30.4879456
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3759308, 39.3700027
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7923279, 30.8049946
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.3722916, 45.3893051
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -29.0525780, 29.0689240
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.8543777, 30.8646355
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5478401, 29.5561905
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.2116394, 46.2228546
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7537766, 39.7588806
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.5073395, 41.5220261
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6912537, 35.7029228
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5273323, 29.5361481
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1303864, 30.1312485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2912134, upper bound: 24.3418859
time: 41.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2999863, upper bound: 24.3331652
time: 41.94 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7893105, 33.7926407
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7913094, 23.7948570
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.5147018, 20.5180397
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6585693, 23.6590080
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4168015, 24.4170341
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7691841, 20.7703438
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2729759, 29.2615585
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.1078453, 20.1080818
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0488434, 28.0580597
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6722870, 30.6716537
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2146149, 37.2142029
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6149445, 27.6163368
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.2208824, 26.2110138
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4311295, 43.4186287
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.6108704, 39.6167374
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9802475, 31.9819756
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3381042, 34.3433952
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1493835, 36.1461143
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4176331, 41.4245834
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.7036667, 30.7058449
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9439392, 31.9458351
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4699402, 33.4765549
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6348038, 27.6341362
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6936951, 28.7013588
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5421753, 33.5458527
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2866974, 34.2947502
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.6902695, 36.6986427
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7829208, 26.7892036
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2382431, 29.2423096
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1654739, 26.1653061
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4879456, 30.4841843
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3700104, 39.3759384
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.8049927, 30.7923317
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.3893204, 45.3722916
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -29.0689201, 29.0525742
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.8646393, 30.8543739
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5561867, 29.5478363
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.2228546, 46.2116470
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7588730, 39.7537918
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.5220184, 41.5073318
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7029266, 35.6912575
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5361443, 29.5273323
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1312408, 30.1303864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1661

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3331652, upper bound: 24.2999863
time: 56.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3418859, upper bound: 24.2912134
time: 41.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 100.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 100.58
Output dim: 17, lower bound: -24.2912134, upper bound: 24.3418859
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 100.58
Output dim: 17, lower bound: -24.2999863, upper bound: 24.3331652
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 100.58
Output dim: 17, lower bound: -24.3331652, upper bound: 24.2999863
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 100.58
Output dim: 17, lower bound: -24.3418859, upper bound: 24.2912134

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7596855, 33.7511940
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7615738, 23.7531471
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4899445, 20.4816418
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6353989, 23.6311684
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4090500, 24.4074631
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7461243, 20.7404289
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2547340, 29.2664833
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0795021, 20.0747604
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0191956, 28.0032272
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6708755, 30.6714897
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2219315, 37.2300987
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6076584, 27.6064568
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1975861, 26.2095261
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4153214, 43.4325714
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5898972, 39.5804520
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9864655, 31.9860153
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3306847, 34.3244286
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1547546, 36.1566887
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4111099, 41.3972473
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6792603, 30.6811218
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9442215, 31.9424210
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4766617, 33.4700241
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5999832, 27.6051712
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7029190, 28.6952362
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5338593, 33.5314064
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2800903, 34.2739792
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7147751, 36.7026901
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7787704, 26.7710266
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2372780, 29.2335548
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1354790, 26.1395226
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4557915, 30.4639816
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3584900, 39.3545456
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7738380, 30.7896004
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2971649, 45.3245621
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9780922, 29.0049706
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7889938, 30.8090019
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5037766, 29.5169449
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1490021, 46.1683121
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7320557, 39.7386780
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.4142303, 41.4424973
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6783905, 35.6931915
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5118942, 29.5239334
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1331177, 30.1339874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2608846, upper bound: 24.3407660
time: 42.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2904821, upper bound: 24.3115478
time: 38.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7545280, 33.7563553
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7566910, 23.7580261
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4849815, 20.4866028
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6316071, 23.6349640
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4076996, 24.4088135
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7415886, 20.7449646
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2550697, 29.2661552
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0750008, 20.0792618
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0124359, 28.0099792
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6708603, 30.6715088
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2296906, 37.2223396
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6078568, 27.6062622
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1996613, 26.2074547
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4200745, 43.4278183
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5863113, 39.5840340
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9877472, 31.9847336
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3297157, 34.3254013
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1534119, 36.1580353
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4042130, 41.4041367
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6833038, 30.6770782
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9443130, 31.9423294
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4766464, 33.4700394
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6044998, 27.6006470
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7029037, 28.6952477
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5350876, 33.5301781
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2820358, 34.2720337
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7110672, 36.7063980
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7773209, 26.7724724
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2376289, 29.2332115
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1393547, 26.1356430
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4602242, 30.4595490
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3604889, 39.3525620
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7769356, 30.7864971
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.3075409, 45.3141785
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9886208, 28.9944420
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7987442, 30.7992516
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5085907, 29.5121307
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1571198, 46.1602020
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7335815, 39.7371368
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.4277954, 41.4289246
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6815262, 35.6900558
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5151215, 29.5207081
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1331253, 30.1339722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2697906, upper bound: 24.3320448
time: 42.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2992547, upper bound: 24.3027325
time: 57.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7563591, 33.7545242
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7580261, 23.7566910
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4866028, 20.4849815
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6349640, 23.6316071
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4088135, 24.4076996
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7449646, 20.7415905
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2661629, 29.2550659
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0792618, 20.0750008
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0099792, 28.0124397
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6715088, 30.6708603
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2223434, 37.2296906
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6062546, 27.6078529
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.2074509, 26.1996613
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4278183, 43.4200745
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5840302, 39.5863190
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9847336, 31.9877472
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3253975, 34.3297157
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1580353, 36.1534195
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.4041367, 41.4042130
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6770782, 30.6833038
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9423294, 31.9443169
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4700394, 33.4766388
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6006546, 27.6045036
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6952438, 28.7029037
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5301743, 33.5350838
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2720337, 34.2820358
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7063980, 36.7110634
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7724686, 26.7773170
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2332191, 29.2376213
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1356468, 26.1393566
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4595528, 30.4602242
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3525543, 39.3604813
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7864952, 30.7769394
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.3141785, 45.3075409
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9944420, 28.9886246
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7992477, 30.7987404
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5121307, 29.5085907
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1602020, 46.1571121
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7371368, 39.7335815
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.4289246, 41.4278030
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6900558, 35.6815262
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5207062, 29.5151196
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1339722, 30.1331253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3027325, upper bound: 24.2992547
time: 44.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3320448, upper bound: 24.2697906
time: 62.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7511940, 33.7596855
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7531509, 23.7615700
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4816437, 20.4899426
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6311646, 23.6354027
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4074631, 24.4090500
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7404289, 20.7461262
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2664833, 29.2547417
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0747604, 20.0795021
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -28.0032272, 28.0191956
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6714935, 30.6708755
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2300949, 37.2219315
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6064529, 27.6076584
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.2095261, 26.1975861
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.4325714, 43.4153214
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5804520, 39.5898972
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9860153, 31.9864655
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3244286, 34.3306885
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1566925, 36.1547623
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3972549, 41.4111023
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6811218, 30.6792603
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9424210, 31.9442253
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4700241, 33.4766541
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.6051712, 27.5999794
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6952362, 28.7029152
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5314102, 33.5338593
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2739792, 34.2800903
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7026901, 36.7147751
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7710342, 26.7787628
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2335548, 29.2372818
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1395226, 26.1354771
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4639854, 30.4557915
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3545380, 39.3584900
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7896004, 30.7738342
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.3245544, 45.2971649
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -29.0049706, 28.9780960
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.8089981, 30.7889900
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.5169449, 29.5037766
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1683197, 46.1489944
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7386780, 39.7320480
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.4425049, 41.4142303
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6931915, 35.6783905
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5239334, 29.5118942
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1339874, 30.1331177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3115478, upper bound: 24.2904821
time: 41.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3407660, upper bound: 24.2608846
time: 41.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 85.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.2608846, upper bound: 24.3407660
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.2904821, upper bound: 24.3115478
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.2697906, upper bound: 24.3320448
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.2992547, upper bound: 24.3027325
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.3027325, upper bound: 24.2992547
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.3320448, upper bound: 24.2697906
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.3115478, upper bound: 24.2904821
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 85.11
Output dim: 17, lower bound: -24.3407660, upper bound: 24.2608846

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7520790, 33.7420883
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7425079, 23.7307816
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4759903, 20.4656773
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6084900, 23.5988464
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4042854, 24.4020462
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7309914, 20.7223263
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2517395, 29.2644882
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0758324, 20.0707684
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9786339, 27.9553452
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6834564, 30.6862869
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2369232, 37.2480049
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5916824, 27.5911560
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1708908, 26.1860542
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3484573, 43.3750076
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5903854, 39.5783463
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9657593, 31.9616013
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3348274, 34.3287010
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1464920, 36.1496124
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3810654, 41.3628006
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6592560, 30.6631012
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9446564, 31.9428520
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4799118, 33.4729691
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5918198, 27.5981674
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7121353, 28.7032509
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5370102, 33.5342865
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2913132, 34.2837105
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7272949, 36.7104263
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7575989, 26.7468185
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2440491, 29.2394676
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1243019, 26.1300774
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4394035, 30.4502831
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3522797, 39.3486557
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7498207, 30.7693462
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2545013, 45.2872925
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9735794, 29.0011177
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7678528, 30.7904587
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4725990, 29.4897652
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1029053, 46.1290207
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7290726, 39.7359390
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3375092, 41.3773575
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6955261, 35.7129784
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5232315, 29.5381889
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1314926, 30.1319427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2465401, upper bound: 24.3399359
time: 43.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2582981, upper bound: 24.3144891
time: 46.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7469139, 33.7472496
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7376404, 23.7356567
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4710274, 20.4706383
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6046982, 23.6026421
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4029350, 24.4033966
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7264557, 20.7268620
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2520676, 29.2641640
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0713310, 20.0752697
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9718742, 27.9620972
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6834412, 30.6863098
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2446747, 37.2402420
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5918808, 27.5909576
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1729584, 26.1839790
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3532104, 43.3702583
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5867996, 39.5819283
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9670334, 31.9603195
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3338509, 34.3296738
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1451492, 36.1509552
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3741837, 41.3696823
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6632996, 30.6590500
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9447479, 31.9427605
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4798965, 33.4729843
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5963440, 27.5936432
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7121277, 28.7032623
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5382385, 33.5330582
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2932663, 34.2817650
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7235870, 36.7141342
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7561569, 26.7482643
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2443848, 29.2391243
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1281776, 26.1261997
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4438362, 30.4458504
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3542786, 39.3466644
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7529335, 30.7662430
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2648773, 45.2769165
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9841080, 28.9905853
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7776108, 30.7807045
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4774132, 29.4849510
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1110077, 46.1209106
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7306137, 39.7343979
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3510895, 41.3637848
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6986542, 35.7098389
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5264587, 29.5349617
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1315079, 30.1319351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2522017, upper bound: 24.3312113
time: 59.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2672511, upper bound: 24.3112810
time: 44.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7472496, 33.7469177
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7356567, 23.7376328
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4706383, 20.4710274
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.6026382, 23.6046982
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4034004, 24.4029388
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7268600, 20.7264576
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2641602, 29.2520676
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0752678, 20.0713310
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9620934, 27.9718781
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6863098, 30.6834412
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2402420, 37.2446823
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5909653, 27.5918732
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1839752, 26.1729622
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3702621, 43.3532104
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5819321, 39.5867996
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9603195, 31.9670334
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3296700, 34.3338509
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1509628, 36.1451530
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3696823, 41.3741837
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6590500, 30.6632996
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9427643, 31.9447441
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4729843, 33.4798889
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5936432, 27.5963440
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7032623, 28.7121277
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5330582, 33.5382347
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2817612, 34.2932625
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7141266, 36.7235870
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7482681, 26.7561531
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2391205, 29.2443886
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1262016, 26.1281776
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4458504, 30.4438324
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3466644, 39.3542786
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7662392, 30.7529297
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2769165, 45.2648849
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9905853, 28.9841042
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7807083, 30.7776070
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4849510, 29.4774094
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1209106, 46.1110153
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7343979, 39.7306137
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3637848, 41.3510818
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7098389, 35.6986580
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5349655, 29.5264568
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1319351, 30.1315079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3112810, upper bound: 24.2672511
time: 38.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3312113, upper bound: 24.2522017
time: 36.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7420921, 33.7520752
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7307739, 23.7425117
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4656754, 20.4759903
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5988464, 23.6084938
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4020500, 24.4042892
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7223244, 20.7309933
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2644882, 29.2517433
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0707664, 20.0758343
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9553413, 27.9786339
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6862869, 30.6834564
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2480011, 37.2369232
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5911484, 27.5916748
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1860504, 26.1708908
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3750076, 43.3484573
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5783463, 39.5903816
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9616013, 31.9657593
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3287010, 34.3348236
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1496048, 36.1464958
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3628006, 41.3810654
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6630936, 30.6592560
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9428558, 31.9446526
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4729691, 33.4799042
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5981674, 27.5918198
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7032471, 28.7121391
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5342865, 33.5370102
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2837143, 34.2913170
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7104187, 36.7272949
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7468185, 26.7575989
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2394714, 29.2440453
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1300774, 26.1242981
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4502831, 30.4393997
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3486481, 39.3522873
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7693520, 30.7498245
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2872925, 45.2545090
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -29.0011139, 28.9735756
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7904587, 30.7678528
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4897652, 29.4725952
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.1290131, 46.1028976
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7359390, 39.7290726
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3773651, 41.3375168
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7129822, 35.6955185
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5381851, 29.5232315
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1319427, 30.1314964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3144891, upper bound: 24.2582981
time: 43.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3399359, upper bound: 24.2465401
time: 17.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 62.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.2465401, upper bound: 24.3399359
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.2582981, upper bound: 24.3144891
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.2522017, upper bound: 24.3312113
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.2672511, upper bound: 24.3112810
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.3112810, upper bound: 24.2672511
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.3312113, upper bound: 24.2522017
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.3144891, upper bound: 24.2582981
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 62.99
Output dim: 17, lower bound: -24.3399359, upper bound: 24.2465401

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7388611, 33.7263184
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7223129, 23.7065659
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4597931, 20.4465313
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5917587, 23.5792694
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3993607, 24.3967667
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7223206, 20.7120018
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2387009, 29.2531471
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0564117, 20.0481300
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9474831, 27.9186401
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6849670, 30.6876717
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2468109, 37.2598381
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6007118, 27.5987778
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1514664, 26.1695099
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3205032, 43.3528786
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5747147, 39.5600662
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9715271, 31.9681320
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3192444, 34.3106079
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1436577, 36.1469345
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3615799, 41.3394165
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6499939, 30.6547127
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9446640, 31.9428673
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4870911, 33.4788170
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5644646, 27.5752373
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7196503, 28.7095032
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5368881, 33.5341721
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2870026, 34.2795639
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7406540, 36.7210121
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7432480, 26.7302189
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2458420, 29.2411499
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1029930, 26.1122169
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4274216, 30.4398537
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3504944, 39.3469391
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7276459, 30.7500629
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2057190, 45.2456589
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9373245, 28.9707527
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7345886, 30.7620468
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4398270, 29.4614334
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0652924, 46.0972824
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7155914, 39.7241364
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2838287, 41.3322601
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6962585, 35.7155800
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5133209, 29.5303764
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1356888, 30.1360893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2341098, upper bound: 24.3393870
time: 43.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2461024, upper bound: 24.3269001
time: 36.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7337036, 33.7314758
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7170639, 23.7114449
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4548340, 20.4514923
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5877609, 23.5830650
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3979721, 24.3981171
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7177887, 20.7165375
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2390213, 29.2528229
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0518265, 20.0526314
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9407310, 27.9253922
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6849518, 30.6876907
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2545624, 37.2518616
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.6008339, 27.5985794
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1535416, 26.1674347
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3252563, 43.3479462
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5711365, 39.5636482
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9728088, 31.9667969
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3182678, 34.3115768
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1423149, 36.1482162
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3546982, 41.3462982
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6540375, 30.6505699
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9447556, 31.9427757
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4870911, 33.4788322
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5689812, 27.5707130
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7195892, 28.7095146
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5381165, 33.5329437
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2889481, 34.2775497
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7369461, 36.7247200
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7417984, 26.7316666
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2461777, 29.2408104
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1068840, 26.1083374
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4318542, 30.4354210
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3524933, 39.3449173
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7307434, 30.7466946
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2160950, 45.2351456
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9478607, 28.9601593
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7443390, 30.7522926
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4446411, 29.4562531
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0734100, 46.0882645
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7171326, 39.7223587
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2973938, 41.3185272
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6993942, 35.7122383
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5165482, 29.5271187
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1356964, 30.1360779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2388116, upper bound: 24.3308983
time: 48.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2515734, upper bound: 24.3195882
time: 47.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7314758, 33.7337036
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7114487, 23.7170639
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4514923, 20.4548340
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5830612, 23.5877647
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3981171, 24.3979759
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7165375, 20.7177868
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2528305, 29.2390213
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0526314, 20.0518265
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9253960, 27.9407272
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6876907, 30.6849518
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2518616, 37.2545662
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5985756, 27.6008301
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1674347, 26.1535416
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3479462, 43.3252563
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5636520, 39.5711365
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9667969, 31.9728050
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3115768, 34.3182678
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1482201, 36.1423073
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3463058, 41.3547058
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6505737, 30.6540337
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9427719, 31.9447594
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4788361, 33.4870872
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5707207, 27.5689850
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7095184, 28.7195854
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5329437, 33.5381165
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2775497, 34.2889481
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7247162, 36.7369461
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7316666, 26.7418022
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2408066, 29.2461815
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1083336, 26.1068764
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4354248, 30.4318504
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3449097, 39.3524933
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7466888, 30.7307415
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2351379, 45.2161026
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9601593, 28.9478569
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7522888, 30.7443390
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4562531, 29.4446411
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0882568, 46.0734024
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7223663, 39.7171326
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3185272, 41.2973938
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7122421, 35.6993942
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5271149, 29.5165520
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1360779, 30.1357002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3195882, upper bound: 24.2515734
time: 40.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3308983, upper bound: 24.2388116
time: 34.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7263184, 33.7388611
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7065659, 23.7223129
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4465332, 20.4597950
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5792694, 23.5917587
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3967667, 24.3993568
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7120018, 20.7223225
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2531509, 29.2386971
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0481300, 20.0564098
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9186440, 27.9474831
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6876678, 30.6849709
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2598419, 37.2468071
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5987740, 27.6007080
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1695099, 26.1514664
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3528748, 43.3205032
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5600662, 39.5747147
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9681320, 31.9715271
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3106079, 34.3192406
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1469383, 36.1436501
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3394241, 41.3615875
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6547089, 30.6499825
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9428635, 31.9446678
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4788208, 33.4870949
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5752373, 27.5644608
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7095032, 28.7196465
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5341721, 33.5368881
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2795639, 34.2870026
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7210083, 36.7406540
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7302170, 26.7432480
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2411499, 29.2458420
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.1122246, 26.1029968
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4398575, 30.4274178
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3469391, 39.3505020
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7500610, 30.7276382
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2456665, 45.2057266
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9707565, 28.9373283
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7620544, 30.7345848
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4614334, 29.4398270
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0972748, 46.0652847
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7241364, 39.7155914
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3322601, 41.2838211
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.7155838, 35.6962509
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5303726, 29.5133247
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1360855, 30.1356888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3269001, upper bound: 24.2461024
time: 27.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3393870, upper bound: 24.2341098
time: 48.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 77.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.2341098, upper bound: 24.3393870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.2461024, upper bound: 24.3269001
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.2388116, upper bound: 24.3308983
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.2515734, upper bound: 24.3195882
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.3195882, upper bound: 24.2515734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.3308983, upper bound: 24.2388116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.3269001, upper bound: 24.2461024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 77.98
Output dim: 17, lower bound: -24.3393870, upper bound: 24.2341098

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7238770, 33.7085724
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7059402, 23.6878738
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4481468, 20.4331608
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5654068, 23.5482025
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4004211, 24.3977623
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7088852, 20.6961842
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2369194, 29.2486877
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0428925, 20.0327129
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9290390, 27.8976555
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6809692, 30.6834831
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2519150, 37.2693634
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5860367, 27.5859985
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1561813, 26.1729012
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3203735, 43.3527527
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5619965, 39.5448227
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9708252, 31.9674988
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3211746, 34.3126984
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1445961, 36.1476784
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3605194, 41.3376541
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6116104, 30.6224747
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9390182, 31.9381981
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4732361, 33.4670296
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5394707, 27.5539703
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7069550, 28.6988525
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5105591, 33.5121193
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2604065, 34.2572212
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7472763, 36.7264938
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7430191, 26.7300568
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2292519, 29.2272491
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0787239, 26.0915089
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4077911, 30.4234085
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3133545, 39.3158188
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7300644, 30.7514191
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2025757, 45.2427826
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9367485, 28.9702225
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7254562, 30.7541580
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4398346, 29.4614449
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0549011, 46.0880814
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7255859, 39.7325592
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2607727, 41.3121185
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6798859, 35.6986275
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5080109, 29.5251503
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1506195, 30.1459351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2260749, upper bound: 24.3388147
time: 49.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2319693, upper bound: 24.3169464
time: 28.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7211151, 33.7113266
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7036209, 23.6901932
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4464264, 20.4348812
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5606918, 23.5529175
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.4003448, 24.3978348
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7065048, 20.6985645
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2342415, 29.2513695
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0409927, 20.0346107
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9264984, 27.9001999
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6807785, 30.6836700
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2563324, 37.2649498
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5879288, 27.5841026
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1548615, 26.1743393
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3203812, 43.3527527
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5594788, 39.5473747
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9709015, 31.9674263
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3212967, 34.3125381
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1443977, 36.1478653
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3598328, 41.3383408
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6177521, 30.6163330
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9399948, 31.9372177
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4752197, 33.4649620
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5432014, 27.5502472
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7090378, 28.6968155
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5148392, 33.5078392
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2646637, 34.2529678
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7461395, 36.7276306
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7430801, 26.7299900
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2319450, 29.2245560
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0822945, 26.0879459
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4109726, 30.4202232
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3193970, 39.3097916
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7289963, 30.7523918
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2028656, 45.2425079
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9367943, 28.9701767
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7266998, 30.7529182
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4398346, 29.4614449
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0561829, 46.0868988
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7240143, 39.7339630
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2636871, 41.3092117
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6792908, 35.6993446
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5080948, 29.5250969
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1455307, 30.1507950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2351709, upper bound: 24.3258888
time: 49.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2443369, upper bound: 24.3110470
time: 41.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7187119, 33.7137299
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.7006912, 23.6927490
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4431839, 20.4381218
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5614166, 23.5519981
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3990479, 24.3991127
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7043495, 20.7007198
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2372475, 29.2483635
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0383072, 20.0372143
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9222870, 27.9044113
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6809464, 30.6835022
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2596817, 37.2613869
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5861588, 27.5858002
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1582947, 26.1708298
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3251266, 43.3478241
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5584412, 39.5484047
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9721069, 31.9661674
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3202057, 34.3136330
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1432381, 36.1489639
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3536224, 41.3445435
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6156540, 30.6183319
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9391098, 31.9380989
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4732361, 33.4670448
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5439949, 27.5494499
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7068939, 28.6988831
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5117798, 33.5108910
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2623520, 34.2552032
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7435608, 36.7302017
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7415695, 26.7315044
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2295876, 29.2269135
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0826073, 26.0876312
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4122238, 30.4189758
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3153381, 39.3138046
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7329865, 30.7480507
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2129517, 45.2322693
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9472771, 28.9596291
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7352142, 30.7444038
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4446564, 29.4562683
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0630188, 46.0791245
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7269592, 39.7307816
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2743378, 41.2983932
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6831055, 35.6952858
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5112381, 29.5218925
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1505356, 30.1459198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2312037, upper bound: 24.3303208
time: 45.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2365968, upper bound: 24.3082401
time: 41.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7137299, 33.7187119
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6927567, 23.7006912
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4381218, 20.4431839
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5520020, 23.5614147
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3991089, 24.3990440
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7007217, 20.7043495
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2483635, 29.2372437
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0372162, 20.0383072
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9044113, 27.9222870
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6835022, 30.6809464
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2613907, 37.2596779
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5857925, 27.5861549
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1708298, 26.1582947
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3478241, 43.3251305
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5484009, 39.5584450
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9661713, 31.9720993
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3136292, 34.3202019
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1489601, 36.1432381
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3445435, 41.3536301
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6183319, 30.6156540
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9380951, 31.9391098
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4670410, 33.4732246
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5494576, 27.5439911
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6988831, 28.7069016
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5108948, 33.5117836
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2552032, 34.2623520
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7302017, 36.7435646
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7314987, 26.7415733
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2269173, 29.2295837
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0876350, 26.0826054
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4189758, 30.4122200
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3138123, 39.3153458
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7480545, 30.7329845
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2322845, 45.2129517
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9596291, 28.9472771
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7444077, 30.7352104
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4562683, 29.4446526
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0791321, 46.0630188
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7307739, 39.7269592
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2983856, 41.2743454
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6952820, 35.6831017
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5218887, 29.5112419
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1459198, 30.1505394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3082401, upper bound: 24.2365968
time: 48.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2260749, upper bound: 24.2312037
time: 18.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7113266, 33.7211151
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6901932, 23.7036171
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4348831, 20.4464245
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5529175, 23.5606918
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3978271, 24.4003525
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6985626, 20.7065048
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2513695, 29.2342377
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0346107, 20.0409927
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9001999, 27.9264984
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6836700, 30.6807823
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2649460, 37.2563324
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5840988, 27.5879288
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1743393, 26.1548615
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3527527, 43.3203773
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5473785, 39.5594749
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9674301, 31.9708939
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3125381, 34.3212967
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1478615, 36.1443977
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3383331, 41.3598328
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6163330, 30.6177521
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9372177, 31.9399948
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4649658, 33.4752159
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5502510, 27.5431976
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6968155, 28.7090416
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5078430, 33.5148392
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2529678, 34.2646637
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7276306, 36.7461357
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7299881, 26.7430878
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2245522, 29.2319412
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0879478, 26.0822906
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4202271, 30.4109688
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3097839, 39.3193893
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7523880, 30.7289925
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2425079, 45.2028503
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9701729, 28.9367943
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7529221, 30.7266960
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4614410, 29.4398384
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0868988, 46.0561829
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7339630, 39.7240143
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3092194, 41.2636795
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6993408, 35.6792984
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5250931, 29.5080986
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1507950, 30.1455307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3110470, upper bound: 24.2443369
time: 42.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3258888, upper bound: 24.2351709
time: 28.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7085724, 33.7238770
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6878738, 23.7059402
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4331589, 20.4481468
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5482025, 23.5654068
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3977661, 24.4004250
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6961823, 20.7088852
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2486916, 29.2369194
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0327148, 20.0428905
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8976593, 27.9290390
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6834793, 30.6809692
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2693634, 37.2519150
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5859909, 27.5860329
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1728973, 26.1561813
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3527527, 43.3203735
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5448227, 39.5619926
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9674988, 31.9708252
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3126984, 34.3211746
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1476784, 36.1445961
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3376617, 41.3605118
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6224747, 30.6116104
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9382019, 31.9390182
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4670410, 33.4732399
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5539742, 27.5394707
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6988525, 28.7069588
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5121155, 33.5105591
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2572250, 34.2604065
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7264938, 36.7472763
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7300491, 26.7430191
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2272530, 29.2292480
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0915108, 26.0787258
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4234085, 30.4077873
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3158264, 39.3133621
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7514191, 30.7300625
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2427826, 45.2025757
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9702187, 28.9367447
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7541580, 30.7254562
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4614410, 29.4398384
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0880737, 46.0549011
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7325592, 39.7255936
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.3121185, 41.2607727
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6986237, 35.6798820
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5251465, 29.5080147
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1459351, 30.1506195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3169464, upper bound: 24.2319693
time: 36.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3388147, upper bound: 24.2260749
time: 15.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 54.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2260749, upper bound: 24.3388147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2319693, upper bound: 24.3169464
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2351709, upper bound: 24.3258888
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2443369, upper bound: 24.3110470
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2312037, upper bound: 24.3303208
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2365968, upper bound: 24.3082401
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.3082401, upper bound: 24.2365968
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.2260749, upper bound: 24.2312037
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.3110470, upper bound: 24.2443369
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.3258888, upper bound: 24.2351709
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.3169464, upper bound: 24.2319693
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 54.81
Output dim: 17, lower bound: -24.3388147, upper bound: 24.2260749

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7154236, 33.6984520
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6897278, 23.6686668
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4385223, 20.4218388
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5473938, 23.5266953
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3960648, 24.3929062
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7008324, 20.6865406
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2390137, 29.2504768
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0309181, 20.0186920
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9069138, 27.8714333
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6866531, 30.6897011
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2679520, 37.2883377
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5818253, 27.5821342
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1484146, 26.1657410
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2880020, 43.3251381
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5577621, 39.5394707
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9740753, 31.9712219
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3155441, 34.3059692
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1455154, 36.1485138
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3475037, 41.3225479
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5948334, 30.6084595
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9355164, 31.9351883
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4715576, 33.4654961
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5215683, 27.5389862
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7054138, 28.6973877
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5056915, 33.5076218
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2538605, 34.2515259
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7510605, 36.7298088
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7350960, 26.7211075
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2287712, 29.2268066
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0613976, 26.0769310
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4007912, 30.4175072
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3034592, 39.3072205
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7249527, 30.7467327
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1872330, 45.2294083
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9356995, 28.9692459
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7137527, 30.7443619
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4206390, 29.4450378
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0325165, 46.0689621
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7229996, 39.7301178
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2249146, 41.2817612
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6777496, 35.6964111
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5093918, 29.5269547
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1562881, 30.1498413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2092656, upper bound: 24.3375115
time: 47.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2247739, upper bound: 24.3220814
time: 45.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7126617, 33.7012100
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6872177, 23.6709900
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4366379, 20.4235611
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5426788, 23.5314102
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3959732, 24.3929787
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6984520, 20.6889210
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2363281, 29.2531624
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0288506, 20.0205917
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9041061, 27.8739738
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6864548, 30.6898003
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2723694, 37.2838211
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5837173, 27.5802422
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1470871, 26.1670799
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2880020, 43.3251381
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5552444, 39.5420227
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9741516, 31.9711494
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3155899, 34.3058052
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1453094, 36.1487007
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3468018, 41.3232346
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6009750, 30.6023178
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9364929, 31.9341736
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4735260, 33.4633446
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5253067, 27.5352173
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7074890, 28.6953354
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5099640, 33.5033417
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2581177, 34.2472725
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7498856, 36.7309494
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7351418, 26.7210407
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2314720, 29.2241096
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0649529, 26.0733204
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4039650, 30.4143219
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3094864, 39.3011856
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7238846, 30.7475243
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1875076, 45.2291260
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9357529, 28.9691849
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7149887, 30.7430496
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4206390, 29.4450378
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0337982, 46.0676804
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7214127, 39.7315216
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2278290, 41.2785339
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6771240, 35.6971321
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5094833, 29.5268536
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1511002, 30.1547050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2183730, upper bound: 24.3245787
time: 33.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2338649, upper bound: 24.3091450
time: 52.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7102661, 33.7036133
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6845093, 23.6735458
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4335594, 20.4267998
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5434036, 23.5304909
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3946915, 24.3942566
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6962967, 20.6910763
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2393341, 29.2501526
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0263367, 20.0231934
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9001694, 27.8781853
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6866226, 30.6897125
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2757111, 37.2803574
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5819473, 27.5819397
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1505203, 26.1636696
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2927551, 43.3204727
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5542145, 39.5430527
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9753571, 31.9698906
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3145676, 34.3069000
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1441422, 36.1497993
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3406067, 41.3294296
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5988770, 30.6043167
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9356079, 31.9350891
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4715424, 33.4655037
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5261002, 27.5344658
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7053528, 28.6974182
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5069122, 33.5063934
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2558136, 34.2495079
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7473526, 36.7335167
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7336464, 26.7225552
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2291145, 29.2264671
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0652733, 26.0730534
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4052162, 30.4130745
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3054428, 39.3051910
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7278671, 30.7433643
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1976089, 45.2189026
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9462357, 28.9586525
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7235031, 30.7346077
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4254532, 29.4400330
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0406189, 46.0601883
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7243576, 39.7283173
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2384796, 41.2680283
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6809845, 35.6930695
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5126266, 29.5236969
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1562042, 30.1498337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2144076, upper bound: 24.3290120
time: 43.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2298958, upper bound: 24.3135896
time: 52.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7012100, 33.7126656
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6709900, 23.6872177
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4235611, 20.4366398
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5314102, 23.5426807
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3929825, 24.3959656
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6889191, 20.6984520
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2531586, 29.2363319
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0205917, 20.0288506
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8739700, 27.9041100
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6897964, 30.6864624
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2838211, 37.2723656
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5802383, 27.5837173
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1670761, 26.1470909
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3251419, 43.2880020
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5420227, 39.5552483
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9711456, 31.9741478
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3058090, 34.3155899
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1487045, 36.1453094
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3232422, 41.3468094
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6023102, 30.6009750
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9341736, 31.9364929
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4633484, 33.4735222
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5352249, 27.5252991
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6953354, 28.7074890
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5033417, 33.5099716
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2472687, 34.2581215
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7309494, 36.7498894
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7210426, 26.7351494
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2241096, 29.2314682
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0733147, 26.0649529
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4143257, 30.4039688
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3011856, 39.3094940
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7475357, 30.7238731
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2291336, 45.1875076
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9691849, 28.9357491
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7430496, 30.7149887
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4450378, 29.4206390
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0676727, 46.0337906
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7315140, 39.7214203
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2785492, 41.2278214
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6971283, 35.6771317
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5268478, 29.5094795
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1547012, 30.1511040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3091450, upper bound: 24.2338649
time: 53.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3245788, upper bound: 24.2183730
time: 50.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6984558, 33.7154236
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6686707, 23.6897240
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4218369, 20.4385223
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5266953, 23.5473976
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3929062, 24.3960648
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6865387, 20.7008324
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2504807, 29.2390137
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0186920, 20.0309181
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8714371, 27.9069176
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6897049, 30.6866493
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2883377, 37.2679520
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5821381, 27.5818214
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1657410, 26.1484108
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3251419, 43.2879982
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5394669, 39.5577660
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9712219, 31.9740753
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3059692, 34.3155441
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1485138, 36.1455116
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3225403, 41.3474960
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6084595, 30.5948257
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9351883, 31.9355202
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4654999, 33.4715538
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5389938, 27.5215721
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6973877, 28.7054062
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5076141, 33.5056915
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2515259, 34.2538643
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7298050, 36.7510605
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7211037, 26.7351017
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2268105, 29.2287750
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0769310, 26.0613899
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4175072, 30.4007874
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3072281, 39.3034592
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7467422, 30.7249432
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2294083, 45.1872253
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9692459, 28.9357033
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7443619, 30.7137527
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4450378, 29.4206390
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0689545, 46.0325089
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7301102, 39.7229996
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2817535, 41.2249146
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6964111, 35.6777496
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5269547, 29.5093956
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1498413, 30.1562881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3220814, upper bound: 24.2247739
time: 35.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3375115, upper bound: 24.2092656
time: 49.28 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 87.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2092656, upper bound: 24.3375115
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2247739, upper bound: 24.3220814
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2183730, upper bound: 24.3245787
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2338649, upper bound: 24.3091450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2144076, upper bound: 24.3290120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.2298958, upper bound: 24.3135896
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.3091450, upper bound: 24.2338649
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.3245788, upper bound: 24.2183730
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.3220814, upper bound: 24.2247739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 87.58
Output dim: 17, lower bound: -24.3375115, upper bound: 24.2092656

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7120743, 33.6932373
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6845512, 23.6602173
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4359703, 20.4172859
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5406036, 23.5158367
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3943329, 24.3906212
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6989098, 20.6840115
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2361565, 29.2478180
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0277290, 20.0135250
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9051590, 27.8655014
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6861687, 30.6892204
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2677383, 37.2882004
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5804367, 27.5809784
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1474113, 26.1644058
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2763977, 43.3186264
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5562363, 39.5373764
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9735718, 31.9712753
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3095474, 34.2973099
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1446609, 36.1476593
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3442383, 41.3167953
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5875320, 30.6038475
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9295959, 31.9316406
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4716415, 33.4654503
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5098648, 27.5315437
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7047958, 28.6968307
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4993896, 33.5038261
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2431564, 34.2446098
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7510834, 36.7297630
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7318077, 26.7190628
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2272224, 29.2255478
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0511589, 26.0706749
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3967476, 30.4149704
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2955093, 39.3020782
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7247238, 30.7465515
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1867142, 45.2289810
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9298706, 28.9617462
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7134323, 30.7435913
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4109879, 29.4391594
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0324783, 46.0689468
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7177658, 39.7264709
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2120209, 41.2734375
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6708145, 35.6892242
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5056992, 29.5231800
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1553650, 30.1478424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1916496, upper bound: 24.3337849
time: 46.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2055559, upper bound: 24.3197186
time: 47.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7102051, 33.6951027
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6812706, 23.6634979
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4339676, 20.4192848
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5365372, 23.5199070
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3937759, 24.3911781
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6983032, 20.6846199
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2363548, 29.2476196
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0257492, 20.0155029
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9009857, 27.8696709
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6861687, 30.6892204
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2678146, 37.2881203
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5806656, 27.5807533
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1470757, 26.1647491
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2814789, 43.3135376
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5556717, 39.5379448
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9741211, 31.9707184
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3068848, 34.2999725
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1446533, 36.1476669
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3417358, 41.3192978
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5902176, 30.6011543
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9319763, 31.9292679
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4715195, 33.4655724
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5141220, 27.5272751
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7048492, 28.6967735
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5018997, 33.5013199
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2469482, 34.2408142
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7510147, 36.7298279
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7330589, 26.7178097
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2275200, 29.2252579
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0551338, 26.0667019
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3982506, 30.4134674
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2983170, 39.2992706
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7247543, 30.7465191
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1868057, 45.2288818
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9281998, 28.9634209
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7129822, 30.7440453
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4147568, 29.4353867
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0324936, 46.0689392
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7193375, 39.7248840
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2165985, 41.2688675
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6705627, 35.6894836
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5056229, 29.5232544
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1542816, 30.1489220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2071361, upper bound: 24.3183958
time: 50.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2210618, upper bound: 24.3043069
time: 39.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7093201, 33.6959915
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6820488, 23.6625404
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4340858, 20.4190063
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5358887, 23.5205536
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3942337, 24.3906975
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6965294, 20.6863918
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2334709, 29.2505035
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0256615, 20.0154228
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9023514, 27.8680458
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6859856, 30.6893196
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2721481, 37.2836838
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5823364, 27.5790825
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1460991, 26.1657410
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2763977, 43.3186226
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5537186, 39.5399246
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9736481, 31.9711990
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3095932, 34.2971458
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1444626, 36.1478424
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3435669, 41.3174820
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5936661, 30.5977058
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9305725, 31.9306259
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4736099, 33.4633064
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5135880, 27.5277710
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7068710, 28.6947746
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5036697, 33.4995461
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2474060, 34.2403564
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7499084, 36.7309036
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7318535, 26.7189941
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2299232, 29.2228546
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0547295, 26.0670624
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3999290, 30.4117889
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3015366, 39.2960510
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7236557, 30.7473450
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1869888, 45.2287064
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9299240, 28.9616852
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7146759, 30.7422791
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4109879, 29.4391594
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0337601, 46.0676727
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7161942, 39.7278671
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2149200, 41.2702179
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6701965, 35.6899452
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5057755, 29.5230789
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1501770, 30.1527023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2007081, upper bound: 24.3208305
time: 22.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2146576, upper bound: 24.3067755
time: 52.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7069092, 33.6983948
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6793327, 23.6650963
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4310074, 20.4222469
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5366135, 23.5196342
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3929520, 24.3919716
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6943741, 20.6885471
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2364769, 29.2474937
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0231438, 20.0180283
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8984146, 27.8722572
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6861534, 30.6892319
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2754898, 37.2802238
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5805664, 27.5807800
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1495247, 26.1623306
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2811508, 43.3139610
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5526886, 39.5409584
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9748535, 31.9699402
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.3085785, 34.2982407
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1432877, 36.1489410
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3373718, 41.3236847
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5915756, 30.5997047
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9296875, 31.9315414
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4716263, 33.4654655
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5143814, 27.5270233
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7047348, 28.6968613
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5006180, 33.5025978
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2450943, 34.2425919
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7473755, 36.7334709
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7303581, 26.7205086
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2275658, 29.2252121
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0550423, 26.0667973
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4011803, 30.4105377
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2974930, 39.3000641
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7276535, 30.7431831
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1970901, 45.2184753
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9404068, 28.9511528
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7231903, 30.7338371
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4157944, 29.4341507
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0405960, 46.0601730
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7191238, 39.7246628
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2255859, 41.2597046
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6740570, 35.6858826
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5089188, 29.5199223
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1552811, 30.1478310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1967375, upper bound: 24.3252487
time: 46.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2106896, upper bound: 24.3111936
time: 48.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6959915, 33.7093163
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6625481, 23.6820488
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4190063, 20.4340858
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5205460, 23.5358925
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3906937, 24.3942375
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6863899, 20.6965313
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2504997, 29.2334709
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0154228, 20.0256615
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8680496, 27.9023476
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6893120, 30.6859856
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2836838, 37.2721481
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5790787, 27.5823364
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1657372, 26.1460953
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3186188, 43.2763977
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5399246, 39.5537224
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9711990, 31.9736481
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2971497, 34.3095932
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1478424, 36.1444626
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3174744, 41.3435593
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5976944, 30.5936775
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9306259, 31.9305725
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4633102, 33.4736061
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5277786, 27.5135880
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6947708, 28.7068748
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4995499, 33.5036697
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2403564, 34.2474060
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7309036, 36.7499084
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7189903, 26.7318497
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2228508, 29.2299194
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0670662, 26.0547256
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4117928, 30.3999290
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2960587, 39.3015442
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7473373, 30.7236614
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2287064, 45.1869812
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9616852, 28.9299240
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7422791, 30.7146759
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4391556, 29.4109840
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0676651, 46.0337677
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7278671, 39.7161865
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2702179, 41.2149277
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6899414, 35.6702003
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5230789, 29.5057793
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1527023, 30.1501808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3067755, upper bound: 24.2146576
time: 42.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3208305, upper bound: 24.2007081
time: 41.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6950989, 33.7102051
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6634941, 23.6812744
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4192848, 20.4339676
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5199051, 23.5365391
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3911819, 24.3937798
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6846199, 20.6983032
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2476158, 29.2363548
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0155029, 20.0257511
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8696671, 27.9009857
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6892204, 30.6861687
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2881241, 37.2678146
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5807571, 27.5806618
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1647453, 26.1470718
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3135376, 43.2814827
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5379410, 39.5556717
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9707184, 31.9741249
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2999725, 34.3068848
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1476669, 36.1446533
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3193054, 41.3417435
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6011581, 30.5902214
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9292679, 31.9319725
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4655838, 33.4715080
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5272751, 27.5141296
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6967697, 28.7048492
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5013199, 33.5018959
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2408142, 34.2469482
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7298279, 36.7510147
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7178078, 26.7330551
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2252617, 29.2275162
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0667076, 26.0551338
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4134712, 30.3982468
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2992630, 39.2983246
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7465134, 30.7247620
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2288895, 45.1868057
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9634171, 28.9282036
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7440414, 30.7129822
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4353867, 29.4147568
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0689468, 46.0325012
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7248917, 39.7193451
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2688599, 41.2165909
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6894836, 35.6705627
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5232544, 29.5056210
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1489182, 30.1542892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3043069, upper bound: 24.2210618
time: 42.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3183958, upper bound: 24.2071361
time: 39.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6932373, 33.7120743
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6602135, 23.6845551
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4172859, 20.4359684
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5158310, 23.5406075
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3906174, 24.3943367
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6840096, 20.6989117
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2478294, 29.2361526
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0135269, 20.0277290
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8655014, 27.9051552
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6892204, 30.6861725
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2882004, 37.2677345
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5809784, 27.5804367
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1644020, 26.1474152
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3186264, 43.2763977
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5373764, 39.5562401
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9712753, 31.9735718
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2973099, 34.3095474
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1476593, 36.1446609
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3168030, 41.3442459
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6038437, 30.5875282
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9316406, 31.9295998
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4654617, 33.4716377
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5315475, 27.5098572
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6968307, 28.7047920
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.5038223, 33.4993896
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2446060, 34.2431526
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7297668, 36.7510796
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7190666, 26.7318039
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2255516, 29.2272263
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0706749, 26.0511608
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4149742, 30.3967438
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.3020706, 39.2955093
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7465439, 30.7247295
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2289810, 45.1866989
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9617462, 28.9298744
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7435913, 30.7134323
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4391556, 29.4109840
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0689468, 46.0324860
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7264633, 39.7177658
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2734375, 41.2120132
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6892242, 35.6708183
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5231857, 29.5056953
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1478424, 30.1553650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3197186, upper bound: 24.2055559
time: 44.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3337849, upper bound: 24.1916496
time: 52.39 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 99.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.1916496, upper bound: 24.3337849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2055559, upper bound: 24.3197186
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2071361, upper bound: 24.3183958
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2210618, upper bound: 24.3043069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2007081, upper bound: 24.3208305
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2146576, upper bound: 24.3067755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.1967375, upper bound: 24.3252487
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.2106896, upper bound: 24.3111936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3067755, upper bound: 24.2146576
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3208305, upper bound: 24.2007081
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3043069, upper bound: 24.2210618
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3183958, upper bound: 24.2071361
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3197186, upper bound: 24.2055559
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 99.59
Output dim: 17, lower bound: -24.3337849, upper bound: 24.1916496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.7041168, 33.6900482
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6794167, 23.6581573
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4325371, 20.4158401
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5394287, 23.5150127
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3946915, 24.3884048
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6965141, 20.6827583
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2320976, 29.2483788
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0165520, 20.0090027
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.9022446, 27.8640366
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6786766, 30.6862221
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2637024, 37.2866859
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5760880, 27.5814247
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1444740, 26.1637306
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2763367, 43.3185196
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5521240, 39.5357208
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9667969, 31.9533005
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2909698, 34.2895966
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1348953, 36.1437454
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3425446, 41.3125458
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5859451, 30.6006279
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9284058, 31.9284744
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4703217, 33.4629478
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5032349, 27.5152893
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7037201, 28.6952248
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4961166, 33.4956512
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2392197, 34.2347984
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7459488, 36.7169800
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7294464, 26.7131443
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2259483, 29.2232018
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0478287, 26.0623627
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3946495, 30.4139938
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2937851, 39.2977829
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7239075, 30.7472153
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1820374, 45.2188339
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9228821, 28.9461861
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7061081, 30.7269821
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4042053, 29.4231377
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0288696, 46.0630875
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7091751, 39.7065353
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2068405, 41.2617493
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6697083, 35.6926231
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5053482, 29.5231781
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1539917, 30.1513596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1433

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1810597, upper bound: 24.2869872
time: 42.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1446500, upper bound: 24.3230408
time: 53.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6989517, 33.6952057
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6741982, 23.6630325
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4275742, 20.4208012
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5354385, 23.5188103
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3933029, 24.3897552
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6919746, 20.6872940
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2324333, 29.2480545
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0119667, 20.0135040
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8955002, 27.8707886
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6786613, 30.6862335
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2714691, 37.2787094
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5762253, 27.5812263
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1465874, 26.1616554
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2810898, 43.3138542
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5485764, 39.5393028
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9680786, 31.9519730
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2899933, 34.2905273
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1335297, 36.1450310
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3356628, 41.3194275
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5899887, 30.5964851
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9284973, 31.9283752
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4703064, 33.4629631
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5077515, 27.5107651
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7036667, 28.6952553
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4973450, 33.4944229
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2411652, 34.2327805
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7422409, 36.7206879
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7280045, 26.7145901
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2262840, 29.2228622
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0517197, 26.0584850
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3990822, 30.4095612
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2957687, 39.2957687
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7268295, 30.7438469
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1924133, 45.2083206
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9334106, 28.9355927
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7158585, 30.7172241
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4090271, 29.4181290
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0369873, 46.0543137
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7105331, 39.7047272
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2204056, 41.2480164
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6729584, 35.6892815
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5085754, 29.5199203
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1539154, 30.1513519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1433

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1860905, upper bound: 24.2784745
time: 48.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1497850, upper bound: 24.3145242
time: 25.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6900482, 33.7041168
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6581535, 23.6794167
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4158401, 20.4325352
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5150070, 23.5394306
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3884048, 24.3946877
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6827583, 20.6965122
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2483788, 29.2321014
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0090027, 20.0165520
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8640366, 27.9022446
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6862297, 30.6786842
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2866821, 37.2637024
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5814285, 27.5760956
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1637306, 26.1444778
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.3185196, 43.2763367
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5357208, 39.5521278
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9533005, 31.9667969
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2895966, 34.2909698
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1437454, 36.1348991
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3125458, 41.3425446
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6006241, 30.5859489
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9284744, 31.9284058
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4629517, 33.4703178
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5152893, 27.5032349
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.6952209, 28.7037201
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4956512, 33.4961166
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2347946, 34.2392197
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7169800, 36.7459488
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.7131424, 26.7294483
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2232018, 29.2259445
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0623703, 26.0478306
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.4139977, 30.3946457
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2977829, 39.2937851
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7472153, 30.7239094
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.2188416, 45.1820450
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9461899, 28.9228821
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7269821, 30.7061043
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4231415, 29.4042053
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -46.0630798, 46.0288773
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7065353, 39.7091675
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2617416, 41.2068329
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6926270, 35.6697159
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.5231781, 29.5053501
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1513596, 30.1539917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1433

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3230408, upper bound: 24.1446500
time: 20.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2869872, upper bound: 24.1810597
time: 46.74 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 70.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.1810597, upper bound: 24.2869872
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.1446500, upper bound: 24.3230408
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.1860905, upper bound: 24.2784745
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.1497850, upper bound: 24.3145242
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.3230408, upper bound: 24.1446500
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 70.15
Output dim: 17, lower bound: -24.2869872, upper bound: 24.1810597

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6802177, 33.6623001
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6673965, 23.6440315
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.4166298, 20.3974400
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5393372, 23.5141373
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3754997, 24.3660660
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.7005386, 20.6910172
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2182922, 29.2373009
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0159836, 20.0085487
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8682938, 27.8251953
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6754379, 30.6836014
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2505569, 37.2753639
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5623398, 27.5699501
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1078529, 26.1322479
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.1893234, 43.2400093
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.5098953, 39.4983063
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9573288, 31.9424057
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2597008, 34.2568359
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.0873184, 36.1033478
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.3158112, 41.2812500
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.5931244, 30.6064911
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9162750, 31.9182701
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4719772, 33.4654694
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.4951401, 27.5067673
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7148132, 28.7025108
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4813232, 33.4789009
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2262726, 34.2200127
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7539139, 36.7236938
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.6985092, 26.6781769
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2334518, 29.2278976
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0263138, 26.0443554
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3626022, 30.3864517
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2936172, 39.2959671
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7124100, 30.7373962
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1371002, 45.1674271
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9125748, 28.9338570
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.6901550, 30.7075462
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.3978271, 29.4178238
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -45.9658966, 45.9884644
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7074966, 39.7044754
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.1772766, 41.2265320
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.5993500, 35.6130829
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.4638443, 29.4752998
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1581802, 30.1540680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1259658, upper bound: 24.3092568
time: 19.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1309433, upper bound: 24.3042312
time: 41.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -22.2170067, 15.9221697, -22.2170067, 15.9221697, -33.6623039, 33.6802216
1: -1.9782817, 26.0864773, -1.9782817, 26.0864773, -23.6440353, 23.6673927
2: 1.4512084, 24.3199158, 1.4512084, 24.3199158, -20.3974419, 20.4166298
3: -12.9399357, 14.6602116, -12.9399357, 14.6602116, -23.5141296, 23.5393333
4: -1.8883278, 26.2459927, -1.8883278, 26.2459927, -24.3660698, 24.3754921
5: 0.2500424, 23.3483562, 0.2500424, 23.3483562, -20.6910172, 20.7005367
6: -74.5473022, -34.0382080, -74.5473022, -34.0382080, -29.2373047, 29.2182846
7: -5.0168333, 19.2839737, -5.0168333, 19.2839737, -20.0085487, 20.0159836
8: -8.8913345, 20.5879021, -8.8913345, 20.5879021, -27.8251953, 27.8682938
9: -26.2783585, 10.9488039, -26.2783585, 10.9488039, -30.6836014, 30.6754379
10: -35.4157066, 10.9634399, -35.4157066, 10.9634399, -37.2753601, 37.2505569
11: -25.0228729, 9.9898891, -25.0228729, 9.9898891, -27.5699539, 27.5623360
12: -42.6141624, -5.9567351, -42.6141624, -5.9567351, -26.1322517, 26.1078491
13: -38.5823479, 20.2198486, -38.5823479, 20.2198486, -43.2400131, 43.1893272
14: 1.8075256, 49.0722122, 1.8075256, 49.0722122, -39.4983063, 39.5098953
15: -17.0181236, 19.4940224, -17.0181236, 19.4940224, -31.9424057, 31.9573288
16: -38.1105690, 6.7308168, -38.1105690, 6.7308168, -34.2568398, 34.2596970
17: 9.8335733, 55.0701523, 9.8335733, 55.0701523, -36.1033401, 36.0873260
18: -8.2213860, 38.7790146, -8.2213860, 38.7790146, -41.2812500, 41.3158112
19: -18.1557693, 17.4224701, -18.1557693, 17.4224701, -30.6064911, 30.5931244
20: -16.1534805, 18.7641850, -16.1534805, 18.7641850, -31.9182739, 31.9162712
21: -16.0742836, 19.3295326, -16.0742836, 19.3295326, -33.4654694, 33.4719696
22: -2.4263172, 31.4893456, -2.4263172, 31.4893456, -27.5067673, 27.4951401
23: -14.9147625, 15.3285322, -14.9147625, 15.3285322, -28.7025070, 28.7148170
24: -8.8338499, 29.2860126, -8.8338499, 29.2860126, -33.4788971, 33.4813271
25: 1.1844578, 39.0329094, 1.1844578, 39.0329094, -34.2200165, 34.2262688
26: -16.2669125, 24.3597660, -16.2669125, 24.3597660, -36.7236938, 36.7539139
27: -14.3723707, 21.7519913, -14.3723707, 21.7519913, -26.6781769, 26.6985092
28: -5.5041218, 23.9309998, -5.5041218, 23.9309998, -29.2278976, 29.2334480
29: -6.3224192, 24.2215576, -6.3224192, 24.2215576, -26.0443573, 26.0263157
30: -12.5472050, 22.1382751, -12.5472050, 22.1382751, -30.3864517, 30.3625946
31: -17.2578125, 27.2593174, -17.2578125, 27.2593174, -39.2959671, 39.2936096
32: -70.1920090, -28.9308071, -70.1920090, -28.9308071, -30.7373886, 30.7124119
33: -110.3536148, -44.7731400, -110.3536148, -44.7731400, -45.1674347, 45.1370926
34: -88.7714386, -42.1550407, -88.7714386, -42.1550407, -28.9338531, 28.9125748
35: -66.5837860, -20.9256172, -66.5837860, -20.9256172, -30.7075500, 30.6901588
36: -60.8301048, -16.0824947, -60.8301048, -16.0824947, -29.4178238, 29.3978310
37: -121.4312592, -55.2024155, -121.4312592, -55.2024155, -45.9884644, 45.9658852
38: -70.6947174, -15.6334362, -70.6947174, -15.6334362, -39.7044754, 39.7074890
39: -111.0771561, -44.4287834, -111.0771561, -44.4287834, -41.2265320, 41.1772842
40: -114.8524780, -64.8247833, -114.8524780, -64.8247833, -35.6130829, 35.5993500
41: -87.1923370, -42.5473709, -87.1923370, -42.5473709, -29.4753036, 29.4638481
42: -72.5599213, -35.9532356, -72.5599213, -35.9532356, -30.1540604, 30.1581841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=329, inp2_unstable=329, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1454
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1455
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 106
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 186
type: RSZ, layer: 1, pos: 1295
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 101
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1322
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3042312, upper bound: 24.1309433
time: 45.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3092568, upper bound: 24.1259658
time: 44.80 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 92.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 92.85
Output dim: 17, lower bound: -24.1259658, upper bound: 24.3092568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 92.85
Output dim: 17, lower bound: -24.1309433, upper bound: 24.3042312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 92.85
Output dim: 17, lower bound: -24.3042312, upper bound: 24.1309433
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 92.85
Output dim: 17, lower bound: -24.3092568, upper bound: 24.1259658

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 60.64 + 3328.18 = 3388.82 seconds
