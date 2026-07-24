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
execution time: IAR + RelationalAnalysis = 2.90 + 55.73 = 58.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 17, lower bound: -24.3458132, upper bound: 24.3458132

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1433

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3350776, upper bound: 24.2989479
time: 22.85 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3350776, upper bound: 24.3350776
time: 45.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 68.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 68.91
Output dim: 17, lower bound: -24.3350776, upper bound: 24.2989479
IS_A2, status: Status.UNKNOWN, split count: 1, time: 68.91
Output dim: 17, lower bound: -24.3350776, upper bound: 24.3350776

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.1574936, 15.9007998, -22.1883545, 15.9195738, -33.7330704, 33.7451591
1: -1.9373472, 26.0729809, -1.9585023, 26.0849495, -23.7568932, 23.7663231
2: 1.4948025, 24.3027477, 1.4722965, 24.3177948, -20.4765129, 20.4845715
3: -12.9327774, 14.6468382, -12.9353781, 14.6568604, -23.6465302, 23.6367226
4: -1.7937286, 26.2230606, -1.8425417, 26.2438240, -24.3201752, 24.3491287
5: 0.2574797, 23.3385639, 0.2537322, 23.3448944, -20.7594948, 20.7469082
6: -74.5269775, -34.1109428, -74.5454865, -34.0735130, -29.2305412, 29.2092552
7: -5.0087891, 19.2701893, -5.0142250, 19.2779427, -20.0955734, 20.0891228
8: -8.8410835, 20.5729828, -8.8670254, 20.5859737, -28.0230942, 28.0263939
9: -26.2714500, 10.9240646, -26.2755432, 10.9380817, -30.6564636, 30.6443329
10: -35.3997269, 10.9069738, -35.4134903, 10.9390011, -37.1762085, 37.1639442
11: -25.0025558, 9.8999043, -25.0208740, 9.9456530, -27.5544357, 27.5239983
12: -42.5809784, -6.0482497, -42.6106224, -6.0010104, -26.1578064, 26.1394997
13: -38.5581589, 20.1175098, -38.5805092, 20.1716309, -43.3503418, 43.3325882
14: 1.8569479, 49.0200272, 1.8197412, 49.0476837, -39.5450974, 39.5527039
15: -16.9181786, 19.4702682, -16.9694366, 19.4911404, -31.8872604, 31.9201050
16: -38.0935898, 6.7062793, -38.1025352, 6.7177205, -34.3127289, 34.2911911
17: 9.8815022, 54.9992905, 9.8425827, 55.0352631, -36.0748177, 36.0754738
18: -8.1110525, 38.7525024, -8.1702394, 38.7770157, -41.3268585, 41.3588943
19: -18.1385841, 17.4170647, -18.1488590, 17.4189320, -30.6720810, 30.6860733
20: -16.1342392, 18.7122765, -16.1517410, 18.7391624, -31.8983765, 31.8872604
21: -16.0558090, 19.2897453, -16.0696373, 19.3105106, -33.4254608, 33.4206581
22: -2.3999295, 31.4865837, -2.4144487, 31.4880905, -27.6003799, 27.6153297
23: -14.9013405, 15.3200359, -14.9096127, 15.3253117, -28.6595459, 28.6740303
24: -8.7886734, 29.2755833, -8.8129921, 29.2845135, -33.4934387, 33.5064964
25: 1.2044268, 39.0091248, 1.1912994, 39.0215569, -34.2516327, 34.2497482
26: -16.2385101, 24.3595352, -16.2568741, 24.3584919, -36.6535263, 36.6822052
27: -14.3024244, 21.7376518, -14.3396673, 21.7508087, -26.7219849, 26.7457371
28: -5.4917364, 23.9242821, -5.5000153, 23.9281139, -29.2041740, 29.2147598
29: -6.3014879, 24.1871376, -6.3187923, 24.2050629, -26.1289825, 26.1271458
30: -12.5219212, 22.0428848, -12.5437498, 22.0923405, -30.4213142, 30.3937073
31: -17.2277012, 27.2551956, -17.2450676, 27.2579041, -39.3431320, 39.3611984
32: -70.1748962, -28.9835205, -70.1892929, -28.9556732, -30.7815018, 30.7685547
33: -110.2991486, -44.8021698, -110.3267670, -44.7771759, -45.3512421, 45.3489075
34: -88.7241516, -42.1783180, -88.7490616, -42.1599083, -29.0348854, 29.0434723
35: -66.5470123, -20.9462109, -66.5664902, -20.9285164, -30.8378754, 30.8434677
36: -60.8151398, -16.0904541, -60.8263130, -16.0861092, -29.5473251, 29.5536766
37: -121.3581467, -55.2316093, -121.3978653, -55.2073097, -46.1779022, 46.1626282
38: -70.6630936, -15.6461716, -70.6827240, -15.6360922, -39.7362213, 39.7456665
39: -111.0365448, -44.4560280, -111.0568085, -44.4324722, -41.5020981, 41.5005951
40: -114.7913208, -64.8541489, -114.8237610, -64.8302078, -35.6327744, 35.6243324
41: -87.1539536, -42.5762177, -87.1742935, -42.5551033, -29.5065994, 29.4839478
42: -72.5446777, -36.0196762, -72.5577240, -35.9845276, -30.0775452, 30.0585136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1703

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2742540, upper bound: 24.2889359
time: 38.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3327141, upper bound: 24.2965876
time: 14.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2011108, 15.9203234, -22.2100620, 15.9213467, -33.7619896, 33.7878532
1: -1.9675522, 26.0852852, -1.9735875, 26.0859299, -23.7793961, 23.7938499
2: 1.4627571, 24.3183422, 1.4562628, 24.3191929, -20.4978752, 20.5157509
3: -12.9371920, 14.6582613, -12.9387569, 14.6592951, -23.6535263, 23.6564312
4: -1.8635609, 26.2445354, -1.8774433, 26.2453537, -24.3773727, 24.4053497
5: 0.2524538, 23.3451195, 0.2511058, 23.3468819, -20.7628365, 20.7770271
6: -74.5460892, -34.0566254, -74.5467911, -34.0463409, -29.2733688, 29.2549400
7: -5.0153990, 19.2803497, -5.0162029, 19.2823486, -20.1044464, 20.1042328
8: -8.8779106, 20.5865059, -8.8854485, 20.5872555, -28.0306473, 28.0635757
9: -26.2764511, 10.9430962, -26.2774715, 10.9463272, -30.6681023, 30.6688347
10: -35.4142647, 10.9495945, -35.4150314, 10.9573555, -37.2080002, 37.1960144
11: -25.0215874, 9.9650326, -25.0223026, 9.9789848, -27.6036530, 27.5817070
12: -42.6121712, -5.9802952, -42.6132278, -5.9670725, -26.2231560, 26.1867599
13: -38.5802612, 20.1938686, -38.5813980, 20.2083435, -43.4394073, 43.3572083
14: 1.8145266, 49.0598602, 1.8107672, 49.0668106, -39.6144791, 39.5784187
15: -16.9923134, 19.4923630, -17.0067978, 19.4932823, -31.9534149, 31.9748077
16: -38.0988312, 6.7235756, -38.1054230, 6.7276359, -34.3139191, 34.3350372
17: 9.8390903, 55.0512619, 9.8360720, 55.0618134, -36.1426620, 36.1026306
18: -8.1940269, 38.7763824, -8.2092476, 38.7778625, -41.3844757, 41.4184341
19: -18.1484871, 17.4204159, -18.1525383, 17.4215717, -30.7064514, 30.6916618
20: -16.1524754, 18.7504482, -16.1529961, 18.7581882, -31.9365692, 31.9213600
21: -16.0711823, 19.3190346, -16.0728874, 19.3249741, -33.4646072, 33.4543381
22: -2.4189601, 31.4850845, -2.4230633, 31.4873943, -27.6249847, 27.6232796
23: -14.9113445, 15.3264523, -14.9132433, 15.3276091, -28.7012177, 28.6800957
24: -8.8223648, 29.2848873, -8.8287792, 29.2855091, -33.5191803, 33.5368423
25: 1.1895847, 39.0232811, 1.1867161, 39.0287170, -34.2804871, 34.2654800
26: -16.2579918, 24.3580818, -16.2629051, 24.3590126, -36.6974106, 36.6855316
27: -14.3544483, 21.7505131, -14.3644438, 21.7513466, -26.7560730, 26.7912598
28: -5.5015182, 23.9285622, -5.5029507, 23.9299259, -29.2414131, 29.2266006
29: -6.3197451, 24.2133179, -6.3212166, 24.2178841, -26.1598206, 26.1425991
30: -12.5448418, 22.1131649, -12.5461464, 22.1272488, -30.4796333, 30.4460831
31: -17.2483749, 27.2584743, -17.2536144, 27.2589550, -39.3668671, 39.3666000
32: -70.1904907, -28.9444695, -70.1913147, -28.9368477, -30.8148804, 30.8005638
33: -110.3452225, -44.7753143, -110.3493729, -44.7741394, -45.3616028, 45.4067459
34: -88.7609940, -42.1577988, -88.7668610, -42.1562347, -29.0678062, 29.0801468
35: -66.5776062, -20.9270840, -66.5807343, -20.9263115, -30.8568344, 30.8745499
36: -60.8279610, -16.0846062, -60.8291512, -16.0834351, -29.5638428, 29.5601616
37: -121.4154053, -55.2053795, -121.4243393, -55.2037354, -46.1722412, 46.2270813
38: -70.6882782, -15.6360359, -70.6918030, -15.6345682, -39.7616501, 39.7614975
39: -111.0718918, -44.4309998, -111.0748291, -44.4297028, -41.5073471, 41.5372314
40: -114.8375702, -64.8289108, -114.8457184, -64.8265610, -35.6294708, 35.6981812
41: -87.1838531, -42.5526886, -87.1886444, -42.5497513, -29.4985657, 29.5330544
42: -72.5586395, -35.9691772, -72.5593719, -35.9603004, -30.1271210, 30.1095200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1703

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2742540, upper bound: 24.3254091
time: 49.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3327141, upper bound: 24.3327142
time: 20.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 73.20 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 73.20
Output dim: 17, lower bound: -24.2742540, upper bound: 24.2889359
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 73.20
Output dim: 17, lower bound: -24.3327141, upper bound: 24.2965876
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 73.20
Output dim: 17, lower bound: -24.2742540, upper bound: 24.3254091
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 73.20
Output dim: 17, lower bound: -24.3327141, upper bound: 24.3327142

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.1560516, 15.9005899, -22.1843872, 15.9189272, -33.7319412, 33.6999016
1: -1.9370465, 26.0716209, -1.9577584, 26.0813522, -23.7606964, 23.7619476
2: 1.4950962, 24.3022919, 1.4731119, 24.3165512, -20.4756622, 20.4822674
3: -12.9326668, 14.6432991, -12.9350948, 14.6472054, -23.6142960, 23.6333141
4: -1.7917085, 26.2229462, -1.8372550, 26.2435112, -24.3202896, 24.3449478
5: 0.2577834, 23.3369522, 0.2545648, 23.3405056, -20.7353477, 20.7441101
6: -74.5267639, -34.1124840, -74.5448761, -34.0778465, -29.1646690, 29.2066498
7: -5.0085287, 19.2678719, -5.0135212, 19.2716751, -20.0599174, 20.0848160
8: -8.8398218, 20.5727959, -8.8636045, 20.5855007, -28.0169144, 28.0241051
9: -26.2691860, 10.9237690, -26.2693348, 10.9372683, -30.6533775, 30.6338081
10: -35.3963013, 10.9066525, -35.4040375, 10.9381294, -37.1697388, 37.1457596
11: -25.0023022, 9.8991108, -25.0200806, 9.9436302, -27.5462265, 27.5177155
12: -42.5807877, -6.0504665, -42.6101608, -6.0070338, -26.0735626, 26.1383972
13: -38.5579910, 20.1160030, -38.5798187, 20.1675091, -43.2952957, 43.3301506
14: 1.8584299, 49.0191841, 1.8236971, 49.0457306, -39.4920502, 39.5481491
15: -16.9174576, 19.4700699, -16.9680462, 19.4906120, -31.8846359, 31.8699570
16: -38.0886993, 6.7061119, -38.0892601, 6.7171974, -34.3085327, 34.2578468
17: 9.8822241, 54.9983253, 9.8444948, 55.0327225, -36.0139847, 36.0735054
18: -8.1086435, 38.7524567, -8.1640244, 38.7768402, -41.3236847, 41.3400269
19: -18.1329403, 17.4170666, -18.1334381, 17.4189053, -30.6698837, 30.6190872
20: -16.1340733, 18.7116966, -16.1513329, 18.7378292, -31.8773117, 31.8847275
21: -16.0528431, 19.2896481, -16.0614395, 19.3103180, -33.4204865, 33.4108887
22: -2.3991747, 31.4865379, -2.4123859, 31.4879608, -27.5971832, 27.6064186
23: -14.9000731, 15.3199072, -14.9061623, 15.3249760, -28.6519089, 28.6410179
24: -8.7865562, 29.2755070, -8.8072681, 29.2843990, -33.4913864, 33.4528885
25: 1.2053556, 39.0090599, 1.1938963, 39.0213928, -34.2486343, 34.1961365
26: -16.2355423, 24.3593330, -16.2488441, 24.3580608, -36.6525116, 36.6748428
27: -14.3002415, 21.7375832, -14.3336782, 21.7505722, -26.7189178, 26.7310886
28: -5.4907923, 23.9240513, -5.4973240, 23.9275112, -29.1978226, 29.1992416
29: -6.3001747, 24.1869850, -6.3152752, 24.2046738, -26.1270905, 26.1188717
30: -12.5214262, 22.0425835, -12.5424194, 22.0915394, -30.3781967, 30.3921547
31: -17.2238255, 27.2551918, -17.2344913, 27.2579041, -39.3411713, 39.2780685
32: -70.1747131, -28.9866943, -70.1887970, -28.9639511, -30.7272720, 30.7659340
33: -110.2979813, -44.8025131, -110.3239594, -44.7779961, -45.3440399, 45.3052979
34: -88.7227325, -42.1808434, -88.7451630, -42.1669731, -29.0257950, 29.0487900
35: -66.5458679, -20.9467144, -66.5638199, -20.9298382, -30.8331757, 30.8173828
36: -60.8149948, -16.0921116, -60.8258324, -16.0906525, -29.5339279, 29.5519905
37: -121.3557587, -55.2318802, -121.3913803, -55.2079010, -46.1748962, 46.0492096
38: -70.6628799, -15.6474838, -70.6820450, -15.6396275, -39.7367859, 39.7418976
39: -111.0346985, -44.4561615, -111.0519791, -44.4329529, -41.4995041, 41.4287262
40: -114.7907486, -64.8544006, -114.8222809, -64.8309402, -35.6310425, 35.5945816
41: -87.1533203, -42.5769234, -87.1726227, -42.5571213, -29.5019760, 29.4697075
42: -72.5443802, -36.0219574, -72.5568848, -35.9908524, -30.0700226, 30.0636330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2875928, upper bound: 24.2942186
time: 49.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3303362, upper bound: 24.2942186
time: 49.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.1609612, 15.9188156, -22.1229591, 15.8938036, -33.6889114, 33.6962662
1: -1.9634464, 26.0811958, -1.9631696, 26.0760059, -23.7622490, 23.7794533
2: 1.4688785, 24.3159866, 1.4721487, 24.3108654, -20.4798775, 20.4945679
3: -12.9361839, 14.6297674, -12.9332991, 14.5945911, -23.5825691, 23.6120739
4: -1.8420084, 26.2428837, -1.8275709, 26.2398281, -24.3432884, 24.3467407
5: 0.2540841, 23.3336143, 0.2636633, 23.3209591, -20.7340012, 20.7470055
6: -74.5432587, -34.1037407, -74.5051117, -34.1467476, -29.1748695, 29.1789055
7: -5.0138226, 19.2618370, -4.9942393, 19.2415600, -20.0662041, 20.0639362
8: -8.8694191, 20.5836678, -8.8639145, 20.5793095, -28.0017700, 28.0276680
9: -26.2662163, 10.9397287, -26.2532597, 10.9432659, -30.6517982, 30.6424446
10: -35.3992767, 10.9435005, -35.3812752, 10.9444780, -37.1704483, 37.1417084
11: -25.0185204, 9.9511776, -25.0083275, 9.9486017, -27.5735283, 27.5588799
12: -42.6088943, -6.0348735, -42.5744667, -6.0842628, -26.0978661, 26.0797729
13: -38.5787811, 20.1374855, -38.5617752, 20.0853958, -43.3182373, 43.2839317
14: 1.8271580, 49.0272408, 1.8733683, 48.9964828, -39.5363197, 39.4922028
15: -16.9450665, 19.4901161, -16.9072151, 19.4684868, -31.8803635, 31.8701248
16: -38.0678749, 6.7222805, -38.0361023, 6.7260714, -34.2783813, 34.2617950
17: 9.8457260, 54.9884033, 9.8839512, 54.9299278, -36.0018845, 35.9958954
18: -8.1420135, 38.7756691, -8.0936947, 38.7602043, -41.3137665, 41.3061295
19: -18.1036110, 17.4200706, -18.0543861, 17.4146767, -30.6304703, 30.5743866
20: -16.1507645, 18.7341881, -16.1310196, 18.7225342, -31.8913879, 31.8829956
21: -16.0514202, 19.3172932, -16.0262489, 19.3236256, -33.4303970, 33.3924522
22: -2.4053078, 31.4842644, -2.3919601, 31.4761753, -27.6093750, 27.5954590
23: -14.8758745, 15.3255129, -14.8369188, 15.2948446, -28.6491547, 28.6025276
24: -8.7797089, 29.2840347, -8.7370977, 29.2613602, -33.4411240, 33.4307861
25: 1.2218342, 39.0209160, 1.2553158, 38.9906998, -34.2196198, 34.2076225
26: -16.2391548, 24.3563976, -16.2100430, 24.3551788, -36.6599960, 36.6148300
27: -14.3207722, 21.7497787, -14.2867088, 21.7470016, -26.7168922, 26.7058811
28: -5.4839439, 23.9267635, -5.4634552, 23.9103699, -29.2085266, 29.1787910
29: -6.3130074, 24.2105064, -6.3034534, 24.2078514, -26.1427155, 26.1207638
30: -12.5415230, 22.0688629, -12.5172129, 22.0331535, -30.3826523, 30.3798676
31: -17.1919060, 27.2582645, -17.1333141, 27.2303371, -39.2716980, 39.2406235
32: -70.1878510, -28.9867210, -70.1595459, -29.0285721, -30.7146034, 30.7193642
33: -110.3152542, -44.7791138, -110.2866287, -44.8157692, -45.2971802, 45.3394470
34: -88.7535858, -42.1712303, -88.7484131, -42.1989517, -28.9966240, 29.0263557
35: -66.5638885, -20.9306812, -66.5513763, -20.9473000, -30.8196030, 30.8390617
36: -60.8248978, -16.0910378, -60.8189697, -16.0981846, -29.5438652, 29.5359306
37: -121.3498306, -55.2070007, -121.2826920, -55.2597733, -46.0457916, 46.0866089
38: -70.6821060, -15.6407318, -70.6776352, -15.6453533, -39.7457657, 39.7434692
39: -111.0322876, -44.4324913, -110.9907913, -44.4678955, -41.4337006, 41.4568558
40: -114.8116760, -64.8300552, -114.7882462, -64.8505402, -35.5801544, 35.6361046
41: -87.1640701, -42.5566788, -87.1452026, -42.5727425, -29.4725647, 29.4968414
42: -72.5572510, -35.9889908, -72.5553131, -36.0081253, -30.0668411, 30.0709686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2291355, upper bound: 24.3230201
time: 48.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2718800, upper bound: 24.3230342
time: 41.17 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.1996231, 15.9201202, -22.2060814, 15.9207716, -33.7608948, 33.7425842
1: -1.9672978, 26.0839329, -1.9728317, 26.0823059, -23.7832108, 23.7894707
2: 1.4630597, 24.3178825, 1.4570725, 24.3179398, -20.4970703, 20.5134373
3: -12.9370813, 14.6547203, -12.9384480, 14.6496601, -23.6212921, 23.6530399
4: -1.8615646, 26.2444420, -1.8722072, 26.2450447, -24.3774719, 24.4011803
5: 0.2527409, 23.3435154, 0.2519164, 23.3424854, -20.7386818, 20.7742424
6: -74.5459137, -34.0582008, -74.5461731, -34.0506554, -29.2074966, 29.2523270
7: -5.0151243, 19.2780113, -5.0154762, 19.2760735, -20.0688057, 20.0999260
8: -8.8766432, 20.5863266, -8.8820152, 20.5868073, -28.0244827, 28.0612106
9: -26.2741928, 10.9428177, -26.2712402, 10.9455309, -30.6649933, 30.6583366
10: -35.4107742, 10.9493141, -35.4056129, 10.9564791, -37.2015457, 37.1777916
11: -25.0213165, 9.9642467, -25.0215282, 9.9769230, -27.5954208, 27.5754051
12: -42.6120148, -5.9825034, -42.6127815, -5.9731202, -26.1388702, 26.1856689
13: -38.5799789, 20.1923428, -38.5807266, 20.2042847, -43.3843307, 43.3547516
14: 1.8159943, 49.0590210, 1.8147564, 49.0648346, -39.5614128, 39.5738678
15: -16.9915447, 19.4921684, -17.0054836, 19.4928093, -31.9508209, 31.9246216
16: -38.0939407, 6.7234197, -38.0921021, 6.7271600, -34.3097153, 34.3016434
17: 9.8397827, 55.0503044, 9.8380518, 55.0592728, -36.0819244, 36.1006012
18: -8.1915550, 38.7763519, -8.2031136, 38.7777328, -41.3813248, 41.3995743
19: -18.1428509, 17.4204025, -18.1371155, 17.4215298, -30.7042313, 30.6247406
20: -16.1523361, 18.7498474, -16.1526146, 18.7568626, -31.9154205, 31.9187851
21: -16.0681915, 19.3189697, -16.0646858, 19.3247585, -33.4596405, 33.4445763
22: -2.4181938, 31.4850368, -2.4209619, 31.4873104, -27.6218300, 27.6143646
23: -14.9100723, 15.3263102, -14.9098082, 15.3272629, -28.6936035, 28.6471100
24: -8.8202429, 29.2848415, -8.8230257, 29.2853794, -33.5171661, 33.4831924
25: 1.1905270, 39.0232162, 1.1893234, 39.0285378, -34.2775421, 34.2119217
26: -16.2550259, 24.3579178, -16.2547951, 24.3586082, -36.6964111, 36.6781006
27: -14.3522539, 21.7504349, -14.3584127, 21.7511444, -26.7529907, 26.7766094
28: -5.5005531, 23.9283409, -5.5002775, 23.9293060, -29.2350349, 29.2109604
29: -6.3184071, 24.2131653, -6.3177023, 24.2174988, -26.1579590, 26.1343632
30: -12.5443830, 22.1128807, -12.5448542, 22.1264877, -30.4365005, 30.4445038
31: -17.2445049, 27.2584610, -17.2430439, 27.2589417, -39.3649750, 39.2834473
32: -70.1903076, -28.9476299, -70.1908340, -28.9450855, -30.7606430, 30.7979355
33: -110.3440399, -44.7755966, -110.3465652, -44.7748718, -45.3544083, 45.3633270
34: -88.7595596, -42.1603737, -88.7628784, -42.1633377, -29.0586586, 29.0853920
35: -66.5764389, -20.9275703, -66.5781021, -20.9276505, -30.8520737, 30.8485565
36: -60.8277855, -16.0863075, -60.8286781, -16.0880241, -29.5504341, 29.5584717
37: -121.4130173, -55.2056007, -121.4178543, -55.2044182, -46.1693039, 46.1136971
38: -70.6880341, -15.6373959, -70.6910858, -15.6381655, -39.7622452, 39.7577515
39: -111.0700912, -44.4311562, -111.0698395, -44.4301949, -41.5048065, 41.4653625
40: -114.8370056, -64.8291245, -114.8443069, -64.8272400, -35.6277237, 35.6684341
41: -87.1832352, -42.5534248, -87.1869812, -42.5517731, -29.4939423, 29.5187836
42: -72.5583649, -35.9715309, -72.5585327, -35.9666214, -30.1195908, 30.1146698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2875928, upper bound: 24.3303285
time: 51.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3303362, upper bound: 24.3303364
time: 35.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 89.22 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.2875928, upper bound: 24.2942186
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.3303362, upper bound: 24.2942186
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.2291355, upper bound: 24.3230201
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.2718800, upper bound: 24.3230342
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.2875928, upper bound: 24.3303285
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 89.22
Output dim: 17, lower bound: -24.3303362, upper bound: 24.3303364

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.2355042, 15.9066696, -22.1814880, 15.9140339, -33.8063049, 33.7004814
1: -2.0417190, 26.0723133, -1.9569066, 26.0763798, -23.8610382, 23.7574081
2: 1.4355915, 24.3059578, 1.4739413, 24.3118877, -20.5299950, 20.4815922
3: -12.9610815, 14.6474123, -12.9340401, 14.6417446, -23.6378021, 23.6325397
4: -1.8667214, 26.2285728, -1.8359444, 26.2383537, -24.3936348, 24.3450012
5: 0.2122970, 23.3429375, 0.2554946, 23.3372536, -20.7796097, 20.7439404
6: -74.5397720, -34.0387955, -74.5359802, -34.0784988, -29.1821480, 29.2619438
7: -5.0621195, 19.2732582, -5.0124903, 19.2682190, -20.1106415, 20.0869904
8: -8.9718447, 20.5700760, -8.8622751, 20.5800018, -28.1475525, 28.0115776
9: -26.3366966, 10.9368401, -26.2679176, 10.9314632, -30.7207870, 30.6343002
10: -35.4242287, 10.9240036, -35.4016075, 10.9306421, -37.2262039, 37.1533241
11: -25.0089149, 9.9296179, -25.0121231, 9.9426937, -27.5451851, 27.5701828
12: -42.5803986, -5.9697223, -42.6023560, -6.0078325, -26.0580177, 26.2158775
13: -38.5624237, 20.2629700, -38.5734940, 20.1641846, -43.2849350, 43.4761009
14: 1.7625647, 49.0267563, 1.8277397, 49.0420227, -39.5731583, 39.5478134
15: -17.0129337, 19.4867096, -16.9671383, 19.4852467, -31.9933167, 31.8748169
16: -38.2179604, 6.7075510, -38.0872192, 6.7080593, -34.4416733, 34.2482224
17: 9.8514900, 55.0903664, 9.8547497, 55.0316849, -36.0275917, 36.1581230
18: -8.1931820, 38.7513008, -8.1612444, 38.7706413, -41.4025192, 41.3298645
19: -18.1423569, 17.4837608, -18.1273632, 17.4186153, -30.6567001, 30.7075195
20: -16.1423359, 18.7809925, -16.1471710, 18.7365494, -31.8732452, 31.9634743
21: -16.0780811, 19.3526096, -16.0571976, 19.3096638, -33.4031448, 33.5119743
22: -2.4094162, 31.5748558, -2.4073219, 31.4871979, -27.6029434, 27.6957779
23: -14.9112597, 15.3587742, -14.9021912, 15.3237896, -28.6235580, 28.7229729
24: -8.7982073, 29.3065147, -8.8008146, 29.2831612, -33.4809189, 33.5179825
25: 1.1856728, 39.0718079, 1.2019444, 39.0196190, -34.2208939, 34.3173027
26: -16.2748890, 24.3717842, -16.2450523, 24.3542595, -36.6554260, 36.7261353
27: -14.3946514, 21.7407455, -14.3310976, 21.7465649, -26.8162155, 26.7245102
28: -5.5022850, 23.9713726, -5.4929953, 23.9261856, -29.1811562, 29.2836456
29: -6.3124771, 24.2563267, -6.3095608, 24.2038784, -26.1345444, 26.1845112
30: -12.5468273, 22.1432590, -12.5366335, 22.0903778, -30.3959846, 30.4895706
31: -17.2482662, 27.2893734, -17.2291927, 27.2570534, -39.3248520, 39.3631287
32: -70.1742477, -28.8888779, -70.1790771, -28.9652653, -30.7210312, 30.8554115
33: -110.2983246, -44.6238556, -110.3146744, -44.7796211, -45.3187256, 45.4983139
34: -88.7261658, -42.0513725, -88.7378845, -42.1680565, -29.0180931, 29.1722603
35: -66.5467529, -20.8303375, -66.5571671, -20.9310570, -30.8196259, 30.9392700
36: -60.8179398, -15.9904461, -60.8199654, -16.0916119, -29.5233078, 29.6504707
37: -121.3628616, -55.1472778, -121.3812561, -55.2089920, -46.1589432, 46.1408920
38: -70.6703415, -15.5876637, -70.6734314, -15.6412954, -39.7294769, 39.8038483
39: -111.0359497, -44.3167038, -111.0433350, -44.4336357, -41.4834671, 41.5750427
40: -114.8117676, -64.8284683, -114.8152618, -64.8322678, -35.6842575, 35.5832176
41: -87.1674118, -42.5103760, -87.1647034, -42.5582199, -29.5279999, 29.5217781
42: -72.5712662, -35.9672966, -72.5521011, -35.9920692, -30.0920410, 30.1143723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1781

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2545312
time: 37.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2914528
time: 34.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.1452370, 15.8663177, -22.1163578, 15.8718252, -33.6493950, 33.6345215
1: -1.9587643, 26.0254974, -1.9612107, 26.0526924, -23.7339172, 23.7214241
2: 1.4730024, 24.2666588, 1.4738846, 24.2902222, -20.4528313, 20.4400578
3: -12.9302273, 14.5873737, -12.9308157, 14.5767603, -23.5558243, 23.5622540
4: -1.8352661, 26.1856537, -1.8247223, 26.2158699, -24.3114243, 24.2844315
5: 0.2589488, 23.3026257, 0.2657132, 23.3079834, -20.7128677, 20.7092476
6: -74.4726562, -34.1057243, -74.4754715, -34.1476288, -29.1153030, 29.1515198
7: -5.0078034, 19.2265167, -4.9917078, 19.2267723, -20.0439758, 20.0233994
8: -8.8632336, 20.5244694, -8.8613310, 20.5545597, -27.9647980, 27.9597244
9: -26.2610168, 10.8797493, -26.2510414, 10.9178333, -30.6153641, 30.5695877
10: -35.3884239, 10.9018660, -35.3767471, 10.9270668, -37.1408920, 37.0862389
11: -24.9922199, 9.9470186, -24.9972553, 9.9468431, -27.5335083, 27.5374107
12: -42.5189819, -6.0388193, -42.5369072, -6.0859132, -26.0038910, 26.0367279
13: -38.5001335, 20.1199226, -38.5289688, 20.0780334, -43.2331085, 43.2298050
14: 1.8572140, 48.9899025, 1.8859129, 48.9803696, -39.4830093, 39.4403343
15: -16.9400253, 19.4354630, -16.9050884, 19.4455147, -31.8500595, 31.8080215
16: -38.0596695, 6.6556673, -38.0326080, 6.6980896, -34.2405243, 34.1887627
17: 9.9660444, 54.9851303, 9.9342804, 54.9285583, -35.8722153, 35.9389877
18: -8.1251812, 38.7040253, -8.0866842, 38.7302551, -41.2630157, 41.2251282
19: -18.0736713, 17.4176235, -18.0418549, 17.4136105, -30.5819626, 30.5514183
20: -16.1097775, 18.7258339, -16.1138153, 18.7190113, -31.8383408, 31.8531494
21: -16.0238304, 19.3108826, -16.0146790, 19.3209629, -33.3731232, 33.3619003
22: -2.3540254, 31.4784279, -2.3702230, 31.4737282, -27.5529099, 27.5674973
23: -14.8538589, 15.3193388, -14.8276386, 15.2922392, -28.5969925, 28.5737267
24: -8.7582092, 29.2768536, -8.7280655, 29.2583046, -33.4003677, 33.4072189
25: 1.2544899, 39.0101624, 1.2690372, 38.9861984, -34.1535263, 34.1712990
26: -16.2131882, 24.3423347, -16.1991673, 24.3492298, -36.6141052, 36.5828896
27: -14.3060789, 21.7027073, -14.2805347, 21.7273293, -26.6803894, 26.6505833
28: -5.4584064, 23.9180870, -5.4527369, 23.9066963, -29.1583519, 29.1499405
29: -6.2553425, 24.2051048, -6.2793350, 24.2055664, -26.0797348, 26.0901127
30: -12.4824867, 22.0607262, -12.4924860, 22.0297203, -30.3196487, 30.3471832
31: -17.1645126, 27.2534866, -17.1218243, 27.2283592, -39.2188187, 39.2124557
32: -70.1166992, -28.9944553, -70.1292648, -29.0318069, -30.6375885, 30.6781998
33: -110.2062988, -44.7877121, -110.2411804, -44.8193855, -45.1842651, 45.2842636
34: -88.6867294, -42.1756401, -88.7201920, -42.2007599, -28.9274521, 28.9929848
35: -66.4919891, -20.9346962, -66.5213318, -20.9490528, -30.7440872, 30.8035545
36: -60.7630653, -16.0950127, -60.7920914, -16.0998821, -29.4744720, 29.5023956
37: -121.2447052, -55.2114143, -121.2386322, -55.2617035, -45.9275208, 46.0311890
38: -70.6175842, -15.6460352, -70.6503067, -15.6475773, -39.6692505, 39.7060699
39: -110.9379349, -44.4360008, -110.9513779, -44.4693108, -41.3360596, 41.4125214
40: -114.7771072, -64.8335724, -114.7736664, -64.8520279, -35.5411377, 35.6022987
41: -87.1172638, -42.5621185, -87.1254196, -42.5750351, -29.4282684, 29.4708824
42: -72.5198135, -35.9945564, -72.5392151, -36.0104599, -30.0281143, 30.0486717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1781

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2263412, upper bound: 24.2832111
time: 41.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2263412, upper bound: 24.3200854
time: 54.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.2404442, 15.9248781, -22.1200905, 15.8888702, -33.7632446, 33.6968346
1: -2.0681255, 26.0818520, -1.9623325, 26.0710316, -23.8625488, 23.7748985
2: 1.4093852, 24.3196449, 1.4729774, 24.3061943, -20.5342102, 20.4938641
3: -12.9646120, 14.6338348, -12.9322739, 14.5891495, -23.6060333, 23.6112900
4: -1.9170184, 26.2484932, -1.8262444, 26.2346878, -24.4166336, 24.3467712
5: 0.2086015, 23.3396034, 0.2645707, 23.3177147, -20.7782707, 20.7468262
6: -74.5562897, -34.0299721, -74.4961853, -34.1474266, -29.1923294, 29.2342110
7: -5.0674272, 19.2672310, -4.9932389, 19.2380981, -20.1169052, 20.0661068
8: -9.0014162, 20.5809612, -8.8625536, 20.5737915, -28.1324043, 28.0151520
9: -26.3337536, 10.9528484, -26.2518139, 10.9374542, -30.7192383, 30.6429176
10: -35.4272346, 10.9609156, -35.3788605, 10.9370461, -37.2268829, 37.1493530
11: -25.0251274, 9.9816761, -25.0003662, 9.9477139, -27.5725136, 27.6113281
12: -42.6084785, -5.9541278, -42.5666885, -6.0849938, -26.0823135, 26.1572800
13: -38.5832787, 20.2844810, -38.5554657, 20.0820942, -43.3079300, 43.4298401
14: 1.7312860, 49.0348396, 1.8773193, 48.9927597, -39.6174088, 39.4918365
15: -17.0405426, 19.5067482, -16.9062462, 19.4631023, -31.9890518, 31.8750305
16: -38.1970978, 6.7237206, -38.0340919, 6.7169323, -34.4115028, 34.2521629
17: 9.8150206, 55.0804443, 9.8941584, 54.9288940, -36.0153923, 36.0805054
18: -8.2265186, 38.7744598, -8.0909348, 38.7539787, -41.3926086, 41.2959213
19: -18.1130352, 17.4867706, -18.0482979, 17.4143791, -30.6172791, 30.6628265
20: -16.1590157, 18.8034935, -16.1268330, 18.7212658, -31.8873901, 31.9617462
21: -16.0766087, 19.3802395, -16.0219841, 19.3229885, -33.4129639, 33.4935455
22: -2.4155779, 31.5726128, -2.3869047, 31.4754124, -27.6150818, 27.6848030
23: -14.8871250, 15.3644104, -14.8329268, 15.2936573, -28.6207199, 28.6844673
24: -8.7913790, 29.3150253, -8.7306490, 29.2601414, -33.4306793, 33.4958878
25: 1.2022228, 39.0836563, 1.2633667, 38.9889374, -34.1919327, 34.3287659
26: -16.2784920, 24.3688412, -16.2062359, 24.3514214, -36.6629486, 36.6661186
27: -14.4152060, 21.7529793, -14.2841396, 21.7429924, -26.8142014, 26.6992874
28: -5.4954786, 23.9740772, -5.4591055, 23.9090691, -29.1917839, 29.2632065
29: -6.3253202, 24.2798119, -6.2977524, 24.2070255, -26.1501389, 26.1863937
30: -12.5669146, 22.1694927, -12.5114031, 22.0319557, -30.4004211, 30.4773293
31: -17.2163582, 27.2923985, -17.1279945, 27.2295074, -39.2553864, 39.3256454
32: -70.1873474, -28.8889580, -70.1498718, -29.0299015, -30.7083740, 30.8089218
33: -110.3155670, -44.6006317, -110.2772675, -44.8173790, -45.2720490, 45.5323868
34: -88.7570190, -42.0417480, -88.7411804, -42.1999893, -28.9889717, 29.1498375
35: -66.5648193, -20.8143387, -66.5447311, -20.9485207, -30.8061523, 30.9609184
36: -60.8278732, -15.9893770, -60.8131256, -16.0990982, -29.5332451, 29.6344261
37: -121.3569946, -55.1225052, -121.2725983, -55.2608795, -46.0297470, 46.1782837
38: -70.6897125, -15.5809174, -70.6689835, -15.6469431, -39.7384109, 39.8053970
39: -111.0335617, -44.2929535, -110.9822540, -44.4685745, -41.4176788, 41.6031723
40: -114.8326797, -64.8041534, -114.7812729, -64.8519440, -35.6333084, 35.6247559
41: -87.1781464, -42.4901237, -87.1372986, -42.5738754, -29.4985619, 29.5489159
42: -72.5840912, -35.9342804, -72.5505219, -36.0093498, -30.0888367, 30.1216812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1781

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2690828, upper bound: 24.2832328
time: 52.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2690828, upper bound: 24.3201138
time: 39.91 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.1839199, 15.8676300, -22.1994553, 15.8987970, -33.7213821, 33.6808395
1: -1.9626269, 26.0282249, -1.9708588, 26.0590172, -23.7548943, 23.7314453
2: 1.4671719, 24.2685471, 1.4588084, 24.2973061, -20.4700165, 20.4589272
3: -12.9311380, 14.6123676, -12.9359531, 14.6318150, -23.5945435, 23.6032314
4: -1.8547983, 26.1872215, -1.8693469, 26.2210999, -24.3456268, 24.3388672
5: 0.2576008, 23.3125210, 0.2539659, 23.3295212, -20.7175331, 20.7364922
6: -74.4752808, -34.0601654, -74.5165558, -34.0514755, -29.1479492, 29.2249603
7: -5.0090904, 19.2426834, -5.0129404, 19.2612686, -20.0465965, 20.0593967
8: -8.8704720, 20.5271416, -8.8794394, 20.5620480, -27.9875221, 27.9932518
9: -26.2689857, 10.8828220, -26.2690563, 10.9200897, -30.6285782, 30.5854988
10: -35.3999252, 10.9076614, -35.4010201, 10.9390745, -37.1719894, 37.1223183
11: -24.9950485, 9.9600601, -25.0105095, 9.9752045, -27.5553970, 27.5539818
12: -42.5220566, -5.9864469, -42.5751572, -5.9747615, -26.0449333, 26.1426163
13: -38.5013618, 20.1748199, -38.5479126, 20.1969032, -43.2992020, 43.3006287
14: 1.8461332, 49.0216637, 1.8273611, 49.0487900, -39.5080757, 39.5220337
15: -16.9865494, 19.4375496, -17.0033340, 19.4698162, -31.9205170, 31.8624802
16: -38.0857544, 6.6567755, -38.0886230, 6.6991849, -34.2718964, 34.2286224
17: 9.9601183, 55.0470047, 9.8883657, 55.0578766, -35.9522858, 36.0437698
18: -8.1747894, 38.7047424, -8.1960335, 38.7477837, -41.3305740, 41.3185730
19: -18.1129704, 17.4179554, -18.1245975, 17.4204712, -30.6557541, 30.6017799
20: -16.1113396, 18.7414837, -16.1354122, 18.7533627, -31.8623886, 31.8889847
21: -16.0406055, 19.3125916, -16.0531197, 19.3220730, -33.4023743, 33.4140244
22: -2.3669119, 31.4792061, -2.3992453, 31.4848042, -27.5653610, 27.5864067
23: -14.8880520, 15.3201427, -14.9005260, 15.3246536, -28.6414337, 28.6182785
24: -8.7987309, 29.2776127, -8.8140163, 29.2823944, -33.4763565, 33.4596672
25: 1.2231598, 39.0124741, 1.2030249, 39.0240364, -34.2113953, 34.1756020
26: -16.2290745, 24.3438549, -16.2439461, 24.3526249, -36.6505051, 36.6462326
27: -14.3375587, 21.7033119, -14.3522673, 21.7314644, -26.7165031, 26.7213421
28: -5.4750471, 23.9196205, -5.4895897, 23.9256439, -29.1848679, 29.1821404
29: -6.2607689, 24.2077522, -6.2935829, 24.2152138, -26.0949821, 26.1037235
30: -12.4853497, 22.1047859, -12.5201464, 22.1230774, -30.3735161, 30.4118118
31: -17.2171211, 27.2536964, -17.2315769, 27.2569561, -39.3121262, 39.2553558
32: -70.1191406, -28.9552994, -70.1605835, -28.9483185, -30.6836052, 30.7567596
33: -110.2351227, -44.7842140, -110.3009949, -44.7784958, -45.2414169, 45.3081665
34: -88.6927032, -42.1647682, -88.7346573, -42.1651764, -28.9894791, 29.0520248
35: -66.5045319, -20.9316330, -66.5480728, -20.9293575, -30.7765350, 30.8130455
36: -60.7659492, -16.0902710, -60.8018417, -16.0896454, -29.4810829, 29.5249405
37: -121.3079376, -55.2100182, -121.3738480, -55.2061996, -46.0511017, 46.0583115
38: -70.6234360, -15.6426497, -70.6637955, -15.6403723, -39.6857758, 39.7204208
39: -110.9757538, -44.4346085, -111.0304337, -44.4317322, -41.4071350, 41.4210205
40: -114.8024139, -64.8325958, -114.8296585, -64.8287430, -35.5887680, 35.6346741
41: -87.1364441, -42.5589142, -87.1672668, -42.5540390, -29.4496536, 29.4928284
42: -72.5209198, -35.9770775, -72.5424805, -35.9689674, -30.0808792, 30.0923729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1781

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2847447, upper bound: 24.2905711
time: 58.00 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2847447, upper bound: 24.3274706
time: 41.76 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.2791443, 15.9262094, -22.2031784, 15.9157763, -33.8352509, 33.7431564
1: -2.0719562, 26.0845699, -1.9719758, 26.0773525, -23.8835297, 23.7849121
2: 1.4035671, 24.3215485, 1.4578969, 24.3132896, -20.5514030, 20.5127449
3: -12.9655037, 14.6588039, -12.9374027, 14.6441708, -23.6447830, 23.6522484
4: -1.9365797, 26.2500496, -1.8708904, 26.2398834, -24.4508362, 24.4012108
5: 0.2072577, 23.3494949, 0.2528243, 23.3392410, -20.7829514, 20.7740536
6: -74.5589218, -33.9844513, -74.5372772, -34.0513000, -29.2249603, 29.3076248
7: -5.0687203, 19.2834206, -5.0144458, 19.2726326, -20.1195259, 20.1020851
8: -9.0086670, 20.5836086, -8.8806496, 20.5813084, -28.1551208, 28.0486946
9: -26.3417358, 10.9559221, -26.2698631, 10.9397430, -30.7324028, 30.6588097
10: -35.4386902, 10.9667130, -35.4031219, 10.9489985, -37.2579422, 37.1853600
11: -25.0279045, 9.9947433, -25.0135803, 9.9760342, -27.5943947, 27.6278877
12: -42.6115532, -5.9017591, -42.6049347, -5.9738889, -26.1233521, 26.2631454
13: -38.5845032, 20.3393135, -38.5744171, 20.2009888, -43.3740234, 43.5007324
14: 1.7201405, 49.0666199, 1.8187475, 49.0611191, -39.6425400, 39.5735397
15: -17.0870590, 19.5087814, -17.0044765, 19.4874001, -32.0595093, 31.9294510
16: -38.2232208, 6.7248249, -38.0901070, 6.7180328, -34.4428482, 34.2920074
17: 9.8091488, 55.1423492, 9.8482637, 55.0581970, -36.0954590, 36.1852570
18: -8.2761250, 38.7751884, -8.2003565, 38.7715187, -41.4601898, 41.3893814
19: -18.1522865, 17.4871311, -18.1310234, 17.4212799, -30.6910706, 30.7131538
20: -16.1605606, 18.8191566, -16.1484146, 18.7555847, -31.9113846, 31.9975700
21: -16.0933990, 19.3819122, -16.0604401, 19.3241138, -33.4422607, 33.5456696
22: -2.4284372, 31.5733719, -2.4159422, 31.4865322, -27.6275406, 27.7037201
23: -14.9213343, 15.3652287, -14.9058037, 15.3261042, -28.6652222, 28.7290535
24: -8.8319569, 29.3158226, -8.8165417, 29.2841415, -33.5066681, 33.5483246
25: 1.1709032, 39.0859756, 1.1973495, 39.0267792, -34.2497559, 34.3330460
26: -16.2944736, 24.3703613, -16.2510223, 24.3548241, -36.6993713, 36.7294159
27: -14.4467030, 21.7535954, -14.3558550, 21.7471046, -26.8503036, 26.7700310
28: -5.5120907, 23.9756374, -5.4959316, 23.9280167, -29.2183533, 29.2953796
29: -6.3307409, 24.2824898, -6.3119841, 24.2166576, -26.1653900, 26.1999683
30: -12.5697985, 22.2135429, -12.5390635, 22.1253128, -30.4542847, 30.5419350
31: -17.2690220, 27.2926140, -17.2377491, 27.2580814, -39.3486710, 39.3685074
32: -70.1898117, -28.8498230, -70.1811752, -28.9464035, -30.7544174, 30.8874779
33: -110.3444061, -44.5970650, -110.3372345, -44.7764587, -45.3292389, 45.5563354
34: -88.7630157, -42.0309448, -88.7556152, -42.1644669, -29.0509834, 29.2088585
35: -66.5773621, -20.8112183, -66.5714111, -20.9289017, -30.8385849, 30.9704514
36: -60.8307610, -15.9846554, -60.8227997, -16.0889091, -29.5398521, 29.6569786
37: -121.4201813, -55.1210938, -121.4077072, -55.2054787, -46.1533585, 46.2053757
38: -70.6956177, -15.5775700, -70.6824570, -15.6398020, -39.7549133, 39.8197098
39: -111.0713654, -44.2917023, -111.0613708, -44.4309349, -41.4887695, 41.6116867
40: -114.8579712, -64.8032074, -114.8372726, -64.8285675, -35.6809464, 35.6571083
41: -87.1973267, -42.4869003, -87.1791077, -42.5528755, -29.5199432, 29.5709190
42: -72.5851746, -35.9167862, -72.5537415, -35.9678497, -30.1416016, 30.1654129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 101
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1781

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2905863
time: 41.98 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3274921, upper bound: 24.3274922
time: 20.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 65.17 seconds
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2545312
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2914528
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2263412, upper bound: 24.2832111
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2263412, upper bound: 24.3200854
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2690828, upper bound: 24.2832328
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2690828, upper bound: 24.3201138
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2847447, upper bound: 24.2905711
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.2847447, upper bound: 24.3274706
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.3274921, upper bound: 24.2905863
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 65.17
Output dim: 17, lower bound: -24.3274921, upper bound: 24.3274922

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.2038422, 15.9052572, -22.1106834, 15.9108067, -33.7702026, 33.6864700
1: -2.0288501, 26.0709610, -1.9281900, 26.0734100, -23.8461113, 23.7268372
2: 1.4381053, 24.3025990, 1.4794998, 24.3044777, -20.5193062, 20.4529266
3: -12.9452391, 14.6416588, -12.8985434, 14.6289673, -23.6431274, 23.5891819
4: -1.8642128, 26.2220078, -1.8303370, 26.2237644, -24.3741341, 24.3294525
5: 0.2256064, 23.3404484, 0.2853065, 23.3318329, -20.7539520, 20.7108459
6: -74.5356369, -34.0474510, -74.5267334, -34.0976982, -29.1522827, 29.2422409
7: -5.0600929, 19.2704391, -5.0080051, 19.2618656, -20.1094360, 20.0775394
8: -8.9698296, 20.5548763, -8.8578014, 20.5463333, -28.1096497, 27.9572449
9: -26.3126125, 10.9310474, -26.2139378, 10.9187279, -30.6704102, 30.5736580
10: -35.3992729, 10.9181767, -35.3457375, 10.9176645, -37.1289444, 37.0923080
11: -25.0043430, 9.9135828, -25.0018883, 9.9068184, -27.4901505, 27.5456810
12: -42.5525589, -5.9711084, -42.5400505, -6.0108109, -25.9457512, 26.1502228
13: -38.5380211, 20.2571812, -38.5187149, 20.1515560, -43.2661667, 43.4143486
14: 1.8029661, 49.0247269, 1.9183893, 49.0377731, -39.4931107, 39.4523926
15: -17.0008564, 19.4821930, -16.9405670, 19.4753017, -31.9710007, 31.8638039
16: -38.1975288, 6.7050915, -38.0414963, 6.7026749, -34.4170265, 34.2010498
17: 9.8965731, 55.0886536, 9.9557772, 55.0279465, -35.8848724, 36.0496597
18: -8.1845875, 38.7501831, -8.1424599, 38.7682228, -41.3797226, 41.4245377
19: -18.1390820, 17.4553490, -18.1203690, 17.3548985, -30.5898285, 30.7039146
20: -16.1409397, 18.7519150, -16.1441460, 18.6714802, -31.8076248, 31.9277496
21: -16.0744324, 19.3270264, -16.0493011, 19.2522869, -33.3440399, 33.4809799
22: -2.4071198, 31.5640602, -2.4022274, 31.4629593, -27.5746536, 27.6887245
23: -14.9077644, 15.3306179, -14.8946743, 15.2606211, -28.5575104, 28.6523705
24: -8.7918205, 29.2795658, -8.7868147, 29.2227268, -33.4118805, 33.5321198
25: 1.1903152, 39.0510445, 1.2120266, 38.9730682, -34.1662140, 34.3038712
26: -16.2716141, 24.3443851, -16.2377720, 24.2929268, -36.5903854, 36.6943855
27: -14.3921146, 21.7099724, -14.3256969, 21.6775208, -26.7431259, 26.5523720
28: -5.4998436, 23.9461708, -5.4877081, 23.8697338, -29.1227341, 29.2253952
29: -6.3083744, 24.2480984, -6.3004150, 24.1854897, -26.1111832, 26.1702938
30: -12.5431833, 22.1266193, -12.5284672, 22.0530663, -30.4036865, 30.4632988
31: -17.2411690, 27.2662907, -17.2134342, 27.2052498, -39.2645187, 39.4176025
32: -70.1700287, -28.8934059, -70.1697693, -28.9750443, -30.7303238, 30.8348198
33: -110.2949066, -44.6474915, -110.3071365, -44.8322983, -45.2621460, 45.4844742
34: -88.7241364, -42.0583496, -88.7334366, -42.1832390, -28.9965897, 29.1518860
35: -66.5426636, -20.8403950, -66.5480347, -20.9535217, -30.7846451, 30.9184914
36: -60.8147278, -16.0110149, -60.8128853, -16.1376648, -29.4729004, 29.6105690
37: -121.3536987, -55.1783485, -121.3609161, -55.2786140, -46.0757828, 46.1534920
38: -70.6657410, -15.6145267, -70.6630783, -15.7014294, -39.6611633, 39.8041611
39: -111.0324249, -44.3467102, -111.0357666, -44.5008698, -41.4122086, 41.5419846
40: -114.8079224, -64.8397751, -114.8065338, -64.8575058, -35.6480637, 35.6329346
41: -87.1636124, -42.5348434, -87.1564789, -42.6130676, -29.4682503, 29.4228821
42: -72.5680542, -35.9815407, -72.5450287, -36.0239525, -30.0583115, 30.0879326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=206, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1431

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2226666
time: 47.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2463386
time: 51.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.2302742, 15.9063148, -22.1895943, 15.9759092, -33.8650436, 33.7022476
1: -2.0395508, 26.0720348, -1.9613214, 26.1096268, -23.8884392, 23.7597847
2: 1.4371157, 24.3051720, 1.4698839, 24.3219414, -20.5380287, 20.4851341
3: -12.9584560, 14.6462030, -12.9322309, 14.6838207, -23.6781769, 23.6253815
4: -1.8649962, 26.2273178, -1.8411889, 26.2441483, -24.4037857, 24.3424225
5: 0.2146277, 23.3424053, 0.2529840, 23.3729458, -20.8138313, 20.7429314
6: -74.5389557, -34.0403061, -74.5666046, -34.0747795, -29.1808319, 29.2804260
7: -5.0603676, 19.2723618, -5.0133433, 19.2758446, -20.1134186, 20.0901070
8: -8.9707861, 20.5676727, -8.9045830, 20.5877266, -28.1469879, 28.0520363
9: -26.3327179, 10.9358139, -26.2655048, 10.9966402, -30.7827454, 30.6246910
10: -35.4200325, 10.9230537, -35.4012642, 11.0082016, -37.3005981, 37.1461639
11: -25.0079784, 9.9268131, -25.0470409, 9.9426861, -27.5430489, 27.6007195
12: -42.5761414, -5.9702969, -42.6001968, -5.9383903, -26.1220970, 26.2028427
13: -38.5587769, 20.2614613, -38.5718346, 20.2415619, -43.3610916, 43.4658585
14: 1.7696533, 49.0262375, 1.8195505, 49.1362534, -39.6618767, 39.5401344
15: -17.0108585, 19.4858360, -16.9763279, 19.5125351, -32.0426407, 31.8668518
16: -38.2142487, 6.7071047, -38.0899162, 6.7591639, -34.4934158, 34.2476234
17: 9.8584776, 55.0898590, 9.8534031, 55.1130066, -36.1031799, 36.1408310
18: -8.1911163, 38.7507706, -8.1745749, 38.7716789, -41.4392090, 41.3179550
19: -18.1415176, 17.4793968, -18.2193794, 17.4134331, -30.6438370, 30.7970276
20: -16.1420288, 18.7763901, -16.2250595, 18.7396183, -31.8722458, 32.0385895
21: -16.0771446, 19.3485737, -16.1276512, 19.3097954, -33.3993988, 33.5814285
22: -2.4087377, 31.5729465, -2.4506145, 31.4865952, -27.5995445, 27.7361336
23: -14.9103727, 15.3543320, -14.9916992, 15.3260574, -28.6183319, 28.8097916
24: -8.7965603, 29.3021317, -8.8697672, 29.2780457, -33.4671097, 33.5823402
25: 1.1870489, 39.0683365, 1.1497669, 39.0192490, -34.2201080, 34.3671150
26: -16.2741623, 24.3670998, -16.3274651, 24.3540440, -36.6508102, 36.8030510
27: -14.3937759, 21.7361965, -14.4417953, 21.7417984, -26.7969513, 26.8319740
28: -5.5017338, 23.9672394, -5.5709553, 23.9297047, -29.1807861, 29.3591309
29: -6.3115253, 24.2547493, -6.3367214, 24.2029610, -26.1311722, 26.2079029
30: -12.5458555, 22.1404877, -12.5708885, 22.0980282, -30.3987770, 30.5362968
31: -17.2465267, 27.2857609, -17.2951813, 27.2523556, -39.3139801, 39.4255219
32: -70.1732101, -28.8904610, -70.1897278, -28.9597130, -30.7224503, 30.8633423
33: -110.2975082, -44.6278076, -110.3842468, -44.7801170, -45.3118896, 45.5649261
34: -88.7256622, -42.0541840, -88.7464294, -42.1690903, -29.0176125, 29.1683617
35: -66.5459290, -20.8325310, -66.5748901, -20.9310150, -30.8144073, 30.9507065
36: -60.8173676, -15.9941511, -60.8826370, -16.0945053, -29.5110550, 29.7101898
37: -121.3610229, -55.1521797, -121.4749603, -55.2127686, -46.1420441, 46.2249069
38: -70.6695557, -15.5925760, -70.7508469, -15.6459036, -39.7122498, 39.8602448
39: -111.0349503, -44.3214569, -111.1331711, -44.4377289, -41.4663010, 41.6596146
40: -114.8111038, -64.8301926, -114.8554077, -64.8286819, -35.6869049, 35.5770264
41: -87.1665421, -42.5144348, -87.2374268, -42.5607491, -29.5136719, 29.5979195
42: -72.5705490, -35.9697037, -72.5901031, -35.9881859, -30.0925674, 30.1606407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1431

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2595910
time: 44.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2832592
time: 33.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.1786613, 15.8672838, -22.2075348, 15.9607277, -33.7801285, 33.6825790
1: -1.9603930, 26.0279598, -1.9752846, 26.0922813, -23.7822876, 23.7338562
2: 1.4686799, 24.2677517, 1.4547546, 24.3073235, -20.4780502, 20.4624538
3: -12.9284897, 14.6111460, -12.9341316, 14.6739140, -23.6349297, 23.5960732
4: -1.8530707, 26.1859798, -1.8746026, 26.2268925, -24.3557701, 24.3362999
5: 0.2599416, 23.3119678, 0.2514834, 23.3651962, -20.7517624, 20.7354774
6: -74.4744568, -34.0617332, -74.5472336, -34.0477905, -29.1466560, 29.2434158
7: -5.0073304, 19.2417793, -5.0137911, 19.2688980, -20.0493965, 20.0624962
8: -8.8694172, 20.5247383, -8.9217253, 20.5697632, -27.9868965, 28.0336838
9: -26.2649841, 10.8817883, -26.2666569, 10.9853268, -30.6905746, 30.5759239
10: -35.3957443, 10.9067259, -35.4006500, 11.0166092, -37.2463760, 37.1151581
11: -24.9941273, 9.9572639, -25.0453510, 9.9751892, -27.5532150, 27.5845032
12: -42.5178375, -5.9870100, -42.5730209, -5.9053578, -26.1090088, 26.1295624
13: -38.4976883, 20.1733685, -38.5462189, 20.2742023, -43.3753586, 43.2904053
14: 1.8532553, 49.0211868, 1.8191881, 49.1430283, -39.5967293, 39.5142822
15: -16.9844418, 19.4366188, -17.0125523, 19.4971313, -31.9698334, 31.8544998
16: -38.0820122, 6.6563191, -38.0913277, 6.7502170, -34.3236084, 34.2280197
17: 9.9670954, 55.0465012, 9.8869963, 55.1392479, -36.0278397, 36.0264969
18: -8.1727438, 38.7042427, -8.2093582, 38.7488518, -41.3672485, 41.3066330
19: -18.1121407, 17.4135914, -18.2165852, 17.4153461, -30.6429138, 30.6912766
20: -16.1110382, 18.7368870, -16.2133007, 18.7563763, -31.8613434, 31.9640503
21: -16.0396881, 19.3085575, -16.1235733, 19.3222294, -33.3985977, 33.4835129
22: -2.3662772, 31.4772911, -2.4424562, 31.4841881, -27.5619888, 27.6267281
23: -14.8871307, 15.3156681, -14.9900646, 15.3269548, -28.6362152, 28.7051239
24: -8.7971172, 29.2732353, -8.8829498, 29.2772789, -33.4626007, 33.5240250
25: 1.2245150, 39.0090141, 1.1508656, 39.0236588, -34.2106171, 34.2254143
26: -16.2283611, 24.3391819, -16.3263435, 24.3523674, -36.6459351, 36.7231598
27: -14.3367138, 21.6987419, -14.4629087, 21.7267036, -26.6972656, 26.8288021
28: -5.4744778, 23.9154797, -5.5674944, 23.9291172, -29.1845016, 29.2576332
29: -6.2597904, 24.2061691, -6.3207150, 24.2143173, -26.0916443, 26.1271305
30: -12.4843397, 22.1019917, -12.5543728, 22.1307526, -30.3762741, 30.4585495
31: -17.2153816, 27.2500992, -17.2975426, 27.2522488, -39.3012543, 39.3177109
32: -70.1181259, -28.9568901, -70.1712112, -28.9428272, -30.6850357, 30.7646980
33: -110.2343369, -44.7881470, -110.3706894, -44.7790413, -45.2346344, 45.3747711
34: -88.6922302, -42.1675873, -88.7432251, -42.1662483, -28.9889755, 29.0481224
35: -66.5037537, -20.9338284, -66.5657806, -20.9293900, -30.7713318, 30.8244896
36: -60.7654152, -16.0939980, -60.8645172, -16.0925293, -29.4688110, 29.5846863
37: -121.3060760, -55.2149353, -121.4676056, -55.2099609, -46.0342484, 46.1424255
38: -70.6226654, -15.6475630, -70.7413254, -15.6450758, -39.6685333, 39.7768402
39: -110.9747314, -44.4393501, -111.1203003, -44.4357185, -41.3899307, 41.5056000
40: -114.8017502, -64.8343658, -114.8697510, -64.8250504, -35.5914078, 35.6284561
41: -87.1356049, -42.5629501, -87.2399292, -42.5565109, -29.4353523, 29.5689468
42: -72.5202103, -35.9794846, -72.5804596, -35.9650803, -30.0813904, 30.1386147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1431

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2181991, upper bound: 24.2957568
time: 38.21 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2764460, upper bound: 24.3191640
time: 44.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.2474823, 15.9247952, -22.1323872, 15.9126034, -33.7991867, 33.7291565
1: -2.0591178, 26.0832634, -1.9432330, 26.0743866, -23.8686142, 23.7543678
2: 1.4060688, 24.3181839, 1.4634395, 24.3058720, -20.5406952, 20.4840946
3: -12.9496651, 14.6530733, -12.9019108, 14.6314087, -23.6500702, 23.6088867
4: -1.9340825, 26.2434711, -1.8652878, 26.2252998, -24.4313393, 24.3856506
5: 0.2205682, 23.3470116, 0.2826591, 23.3338127, -20.7572823, 20.7409649
6: -74.5547638, -33.9930916, -74.5280151, -34.0704689, -29.1950912, 29.2879448
7: -5.0666790, 19.2805614, -5.0099516, 19.2662754, -20.1183014, 20.0926418
8: -9.0066385, 20.5684280, -8.8762016, 20.5476189, -28.1172028, 27.9943352
9: -26.3176346, 10.9501495, -26.2158813, 10.9269657, -30.6820221, 30.5981712
10: -35.4138107, 10.9608765, -35.3472214, 10.9360619, -37.1606979, 37.1243553
11: -25.0233326, 9.9787502, -25.0033207, 9.9401340, -27.5393219, 27.6034126
12: -42.5837479, -5.9031553, -42.5425911, -5.9768896, -26.0110893, 26.1974907
13: -38.5600815, 20.3336048, -38.5196342, 20.1884003, -43.3552780, 43.4389725
14: 1.7606144, 49.0645752, 1.9094572, 49.0569077, -39.5625229, 39.4781113
15: -17.0749779, 19.5042801, -16.9779091, 19.4774666, -32.0371323, 31.9185028
16: -38.2027664, 6.7223573, -38.0443993, 6.7126350, -34.4182129, 34.2448807
17: 9.8542070, 55.1406403, 9.9492865, 55.0545006, -35.9527435, 36.0767899
18: -8.2675476, 38.7741089, -8.1815500, 38.7691765, -41.4374237, 41.4841156
19: -18.1490288, 17.4587231, -18.1240368, 17.3575687, -30.6241989, 30.7095757
20: -16.1591759, 18.7900963, -16.1454010, 18.6904907, -31.8457642, 31.9618149
21: -16.0897865, 19.3563461, -16.0525475, 19.2667370, -33.3831940, 33.5146751
22: -2.4261699, 31.5625916, -2.4108486, 31.4622898, -27.5992661, 27.6966629
23: -14.9178343, 15.3370590, -14.8983107, 15.2629356, -28.5991974, 28.6584549
24: -8.8255615, 29.2888489, -8.8025990, 29.2237606, -33.4376450, 33.5624771
25: 1.1755219, 39.0651817, 1.2074757, 38.9802170, -34.1950684, 34.3196182
26: -16.2911739, 24.3429890, -16.2437458, 24.2934647, -36.6343613, 36.6976852
27: -14.4441910, 21.7228088, -14.3504381, 21.6780128, -26.7772293, 26.5978928
28: -5.5096397, 23.9504147, -5.4906340, 23.8715515, -29.1599503, 29.2371368
29: -6.3266010, 24.2742691, -6.3028088, 24.1982803, -26.1420288, 26.1857586
30: -12.5661135, 22.1969032, -12.5308847, 22.0880165, -30.4619904, 30.5157013
31: -17.2619190, 27.2695274, -17.2219582, 27.2063007, -39.2883148, 39.4229736
32: -70.1856308, -28.8543205, -70.1718216, -28.9562244, -30.7637177, 30.8668823
33: -110.3409729, -44.6205978, -110.3297119, -44.8291779, -45.2726517, 45.5424652
34: -88.7610397, -42.0379028, -88.7511444, -42.1796951, -29.0294952, 29.1884995
35: -66.5732880, -20.8213387, -66.5623322, -20.9513988, -30.8035889, 30.9496574
36: -60.8275681, -16.0051918, -60.8157043, -16.1349773, -29.4894180, 29.6170540
37: -121.4110260, -55.1521721, -121.3873444, -55.2750893, -46.0701599, 46.2180023
38: -70.6909561, -15.6044474, -70.6721420, -15.6999826, -39.6866379, 39.8199844
39: -111.0678864, -44.3216896, -111.0537033, -44.4981422, -41.4174576, 41.5785675
40: -114.8541183, -64.8145370, -114.8284988, -64.8538208, -35.6447601, 35.7068481
41: -87.1935349, -42.5113564, -87.1708221, -42.6076813, -29.4602165, 29.4719772
42: -72.5819855, -35.9310112, -72.5466766, -35.9997177, -30.1078415, 30.1389694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=206, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1431

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2588651
time: 35.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2823101
time: 44.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.2738838, 15.9258471, -22.2112656, 15.9777260, -33.8939896, 33.7449036
1: -2.0697536, 26.0843334, -1.9763839, 26.1105957, -23.9109726, 23.7873116
2: 1.4050701, 24.3207455, 1.4538395, 24.3233242, -20.5594254, 20.5162926
3: -12.9628544, 14.6576109, -12.9355974, 14.6862392, -23.6851501, 23.6450863
4: -1.9348693, 26.2488022, -1.8761294, 26.2456932, -24.4609909, 24.3986397
5: 0.2095942, 23.3489456, 0.2503395, 23.3749199, -20.8171577, 20.7730274
6: -74.5580597, -33.9860268, -74.5679016, -34.0476112, -29.2236481, 29.3260803
7: -5.0669627, 19.2825069, -5.0153217, 19.2802391, -20.1222954, 20.1051826
8: -9.0076227, 20.5812206, -8.9229755, 20.5890274, -28.1545067, 28.0891418
9: -26.3377171, 10.9549026, -26.2674961, 11.0049152, -30.7943840, 30.6492348
10: -35.4345055, 10.9657307, -35.4027596, 11.0265522, -37.3323593, 37.1781998
11: -25.0269699, 9.9919682, -25.0484619, 9.9760132, -27.5922127, 27.6584320
12: -42.6072845, -5.9023428, -42.6027908, -5.9044518, -26.1874084, 26.2501144
13: -38.5807991, 20.3378220, -38.5727081, 20.2783203, -43.4501953, 43.4904747
14: 1.7272539, 49.0661087, 1.8105717, 49.1553917, -39.7312164, 39.5658798
15: -17.0849705, 19.5078621, -17.0137081, 19.5146828, -32.1088104, 31.9214783
16: -38.2194824, 6.7243423, -38.0928116, 6.7691021, -34.4945908, 34.2914200
17: 9.8161621, 55.1418381, 9.8468866, 55.1395721, -36.1709976, 36.1679764
18: -8.2741318, 38.7746887, -8.2135906, 38.7725906, -41.4968719, 41.3774796
19: -18.1514206, 17.4827824, -18.2230568, 17.4161091, -30.6782379, 30.8026848
20: -16.1602592, 18.8145943, -16.2263145, 18.7586193, -31.9104080, 32.0726776
21: -16.0925198, 19.3779144, -16.1308956, 19.3242378, -33.4385147, 33.6150932
22: -2.4277987, 31.5714645, -2.4592171, 31.4859200, -27.6241531, 27.7440300
23: -14.9204435, 15.3607674, -14.9953470, 15.3283672, -28.6599884, 28.8158913
24: -8.8303375, 29.3114357, -8.8855171, 29.2790565, -33.4929123, 33.6126633
25: 1.1722546, 39.0825272, 1.1451740, 39.0264320, -34.2490082, 34.3828659
26: -16.2937565, 24.3657055, -16.3334179, 24.3545895, -36.6947479, 36.8063507
27: -14.4458256, 21.7490063, -14.4665222, 21.7423458, -26.8310623, 26.8774986
28: -5.5115108, 23.9715157, -5.5738888, 23.9314823, -29.2179680, 29.3708572
29: -6.3297567, 24.2808838, -6.3391352, 24.2157745, -26.1620255, 26.2233868
30: -12.5687990, 22.2107372, -12.5732889, 22.1329575, -30.4570732, 30.5886726
31: -17.2672729, 27.2890396, -17.3037567, 27.2533684, -39.3377762, 39.4309082
32: -70.1887970, -28.8514080, -70.1917572, -28.9408684, -30.7558365, 30.8954048
33: -110.3435898, -44.6009102, -110.4069061, -44.7770386, -45.3224335, 45.6229401
34: -88.7625427, -42.0337257, -88.7642136, -42.1654930, -29.0505066, 29.2049942
35: -66.5765686, -20.8134003, -66.5891800, -20.9288921, -30.8333817, 30.9818916
36: -60.8301544, -15.9883432, -60.8854904, -16.0918274, -29.5275879, 29.7166901
37: -121.4183273, -55.1260262, -121.5014420, -55.2092400, -46.1364594, 46.2894363
38: -70.6948242, -15.5824347, -70.7599792, -15.6444244, -39.7377472, 39.8761063
39: -111.0702744, -44.2963829, -111.1511230, -44.4349365, -41.4715576, 41.6962204
40: -114.8573303, -64.8049622, -114.8773804, -64.8249512, -35.6836090, 35.6508980
41: -87.1964645, -42.4909058, -87.2517395, -42.5553551, -29.5056000, 29.6470070
42: -72.5845032, -35.9192009, -72.5917206, -35.9639549, -30.1421127, 30.2116508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1431

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2957809
time: 45.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3191910, upper bound: 24.3191911
time: 44.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 92.39 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2226666
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2463386
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2595910
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2832592
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.2181991, upper bound: 24.2957568
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.2764460, upper bound: 24.3191640
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2588651
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3191910, upper bound: 24.2823101
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3190964, upper bound: 24.2957809
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 92.39
Output dim: 17, lower bound: -24.3191910, upper bound: 24.3191911

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 58.62 + 1355.83 = 1414.45 seconds
