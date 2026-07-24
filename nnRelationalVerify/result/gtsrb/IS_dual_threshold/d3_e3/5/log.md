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
execution time: IAR + RelationalAnalysis = 2.90 + 55.59 = 58.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 17, lower bound: -24.3458132, upper bound: 24.3458132

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1781

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3061261, upper bound: 24.3430450
time: 52.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3430450, upper bound: 24.3430450
time: 26.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 79.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 79.53
Output dim: 17, lower bound: -24.3061261, upper bound: 24.3430450
IS_A2, status: Status.UNKNOWN, split count: 1, time: 79.53
Output dim: 17, lower bound: -24.3430450, upper bound: 24.3430450

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.1462440, 15.9189863, -22.1853352, 15.9207268, -33.7828522, 33.7607841
1: -1.9495373, 26.0835075, -1.9654343, 26.0851192, -23.7692795, 23.7849312
2: 1.4567544, 24.3124752, 1.4536998, 24.3165436, -20.4936295, 20.5115776
3: -12.9044485, 14.6474552, -12.9241114, 14.6544590, -23.6175079, 23.6661968
4: -1.8827295, 26.2314014, -1.8858149, 26.2394257, -24.4019051, 24.3979568
5: 0.2798858, 23.3429298, 0.2633648, 23.3458710, -20.7395439, 20.7469940
6: -74.5380936, -34.0574265, -74.5431519, -34.0468292, -29.2634201, 29.2532310
7: -5.0123529, 19.2776127, -5.0147991, 19.2811165, -20.0990944, 20.1073303
8: -8.8868761, 20.5542202, -8.8893127, 20.5726929, -28.0164795, 28.0329132
9: -26.2243500, 10.9360332, -26.2542572, 10.9430199, -30.6149826, 30.6252136
10: -35.3598557, 10.9504871, -35.3907967, 10.9576340, -37.1550522, 37.1187859
11: -25.0126495, 9.9539928, -25.0183105, 9.9738436, -27.5910568, 27.5605087
12: -42.5518494, -5.9597206, -42.5863724, -5.9581366, -26.1698418, 26.1232452
13: -38.5275803, 20.2072048, -38.5578995, 20.2141418, -43.3861160, 43.4289970
14: 1.8982306, 49.0679474, 1.8480568, 49.0701675, -39.5275955, 39.5427361
15: -16.9915771, 19.4840946, -17.0060482, 19.4895153, -31.9772491, 31.9658813
16: -38.0648346, 6.7254457, -38.0901413, 6.7283392, -34.3040695, 34.3265839
17: 9.9346561, 55.0664520, 9.8786440, 55.0684280, -36.0452652, 36.0110016
18: -8.2026005, 38.7766533, -8.2128010, 38.7779160, -41.5282440, 41.4107819
19: -18.1487789, 17.3587437, -18.1524887, 17.3940468, -30.7008438, 30.6375656
20: -16.1504402, 18.6991177, -16.1520920, 18.7351418, -31.9085693, 31.8786697
21: -16.0663681, 19.2721920, -16.0706482, 19.3039322, -33.4402008, 33.4121132
22: -2.4212103, 31.4650803, -2.4239788, 31.4785156, -27.6290283, 27.6077805
23: -14.9072590, 15.2653313, -14.9112635, 15.3003407, -28.6245422, 28.6291313
24: -8.8198652, 29.2256298, -8.8274517, 29.2590923, -33.5577011, 33.4745026
25: 1.1945467, 38.9863586, 1.1890774, 39.0121307, -34.2749100, 34.2336349
26: -16.2596111, 24.2984581, -16.2636414, 24.3323650, -36.6646118, 36.6313248
27: -14.3669271, 21.6829243, -14.3698254, 21.7212219, -26.6255341, 26.7245979
28: -5.4987869, 23.8745251, -5.5016394, 23.9057922, -29.1810913, 29.1809502
29: -6.3132362, 24.2031422, -6.3182487, 24.2133350, -26.1517372, 26.1426277
30: -12.5390396, 22.1010075, -12.5434618, 22.1216316, -30.4667511, 30.5005875
31: -17.2420635, 27.2075062, -17.2507343, 27.2362022, -39.4276581, 39.3128433
32: -70.1826935, -28.9406166, -70.1878357, -28.9353294, -30.8020248, 30.8319187
33: -110.3460922, -44.8258247, -110.3501968, -44.7965851, -45.3990784, 45.3563766
34: -88.7669144, -42.1702271, -88.7693939, -42.1620407, -29.0683670, 29.0672302
35: -66.5746307, -20.9481583, -66.5796661, -20.9357090, -30.8578873, 30.8436890
36: -60.8229752, -16.1285820, -60.8268700, -16.1030540, -29.5284500, 29.5179443
37: -121.4108887, -55.2720413, -121.4220963, -55.2334633, -46.2518921, 46.1560669
38: -70.6843414, -15.6936646, -70.6900635, -15.6602983, -39.7692032, 39.7005768
39: -111.0694885, -44.4960632, -111.0736465, -44.4587898, -41.5096741, 41.4714737
40: -114.8436890, -64.8499603, -114.8486023, -64.8361206, -35.7550354, 35.6691628
41: -87.1841278, -42.6022034, -87.1885834, -42.5718231, -29.4445496, 29.4837341
42: -72.5528870, -35.9851265, -72.5567093, -35.9674759, -30.1067047, 30.0993614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1433

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2592835, upper bound: 24.3322432
time: 34.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2953353, upper bound: 24.3322432
time: 35.87 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.2250938, 15.9841003, -22.2117634, 15.9218407, -33.7986298, 33.8555832
1: -1.9827113, 26.1197224, -1.9760780, 26.0862236, -23.8022194, 23.8272438
2: 1.4471331, 24.3299446, 1.4527118, 24.3191223, -20.5257950, 20.5302944
3: -12.9381447, 14.7022791, -12.9373245, 14.6589985, -23.6537323, 23.7012272
4: -1.8935776, 26.2517853, -1.8865852, 26.2447605, -24.4148941, 24.4276276
5: 0.2475395, 23.3840199, 0.2523708, 23.3478050, -20.7716293, 20.8068733
6: -74.5779724, -34.0344887, -74.5464401, -34.0397339, -29.3015671, 29.2817955
7: -5.0176940, 19.2915916, -5.0150652, 19.2830734, -20.1116447, 20.1113243
8: -8.9336090, 20.5956211, -8.8902721, 20.5854988, -28.1112671, 28.0702629
9: -26.2759171, 11.0140142, -26.2743702, 10.9478188, -30.6660423, 30.7376022
10: -35.4153938, 11.0410004, -35.4115257, 10.9625158, -37.2088776, 37.2904587
11: -25.0577908, 9.9898911, -25.0219231, 9.9870777, -27.6460800, 27.6133881
12: -42.6120377, -5.8872995, -42.6099319, -5.9572973, -26.2224884, 26.2996178
13: -38.5806427, 20.2971535, -38.5786591, 20.2183304, -43.4375687, 43.5239487
14: 1.7993174, 49.1664352, 1.8146429, 49.0716896, -39.6153450, 39.7117157
15: -17.0273781, 19.5213432, -17.0160675, 19.4931011, -31.9802246, 32.0375443
16: -38.1132507, 6.7818451, -38.1068115, 6.7303452, -34.3506012, 34.4029617
17: 9.8322487, 55.1514893, 9.8405581, 55.0696411, -36.1364479, 36.2293053
18: -8.2346840, 38.7800980, -8.2194061, 38.7784958, -41.4216537, 41.4702225
19: -18.2477875, 17.4172764, -18.1549129, 17.4180984, -30.7939377, 30.6915779
20: -16.2313824, 18.7672329, -16.1531677, 18.7596188, -32.0193939, 31.9433098
21: -16.1447182, 19.3296490, -16.0733604, 19.3254852, -33.5406647, 33.4674530
22: -2.4695482, 31.4887314, -2.4256592, 31.4874134, -27.6763763, 27.6326904
23: -15.0042820, 15.3308096, -14.9138756, 15.3240566, -28.7819672, 28.6899414
24: -8.9027863, 29.2809334, -8.8321800, 29.2816124, -33.6078796, 33.5297508
25: 1.1322803, 39.0325546, 1.1857948, 39.0294380, -34.3381424, 34.2875099
26: -16.3493214, 24.3595657, -16.2661781, 24.3551006, -36.7732544, 36.6917343
27: -14.4830132, 21.7472267, -14.3714590, 21.7474174, -26.9051132, 26.7784290
28: -5.5820341, 23.9345016, -5.5035505, 23.9268913, -29.3148155, 29.2389908
29: -6.3495436, 24.2206440, -6.3214064, 24.2199497, -26.1893692, 26.1626148
30: -12.5814619, 22.1459427, -12.5462036, 22.1355019, -30.5397377, 30.4957581
31: -17.3238049, 27.2546120, -17.2560959, 27.2557201, -39.4355392, 39.3622742
32: -70.2026596, -28.9252739, -70.1910477, -28.9324341, -30.8305740, 30.8240738
33: -110.4231720, -44.7735939, -110.3528519, -44.7769852, -45.4795685, 45.4061584
34: -88.7799911, -42.1560860, -88.7708817, -42.1577873, -29.0848312, 29.0882301
35: -66.6014862, -20.9256058, -66.5829620, -20.9278469, -30.8901138, 30.8734627
36: -60.8927307, -16.0853367, -60.8295326, -16.0862083, -29.6280861, 29.5561104
37: -121.5250702, -55.2061768, -121.4294586, -55.2073784, -46.3233337, 46.2223206
38: -70.7721481, -15.6380892, -70.6939163, -15.6383371, -39.8252869, 39.7516632
39: -111.1669769, -44.4328308, -111.0761032, -44.4335442, -41.6273041, 41.5255890
40: -114.8925552, -64.8211212, -114.8517914, -64.8265228, -35.6991653, 35.7079887
41: -87.2650375, -42.5498428, -87.1915207, -42.5513878, -29.6195679, 29.5291424
42: -72.5979156, -35.9493599, -72.5592194, -35.9556541, -30.1793823, 30.1336594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1433

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2953354, upper bound: 24.2962088
time: 55.52 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3322431, upper bound: 24.3322431
time: 39.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 97.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 97.21
Output dim: 17, lower bound: -24.2592835, upper bound: 24.3322432
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 97.21
Output dim: 17, lower bound: -24.2953353, upper bound: 24.3322432
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 97.21
Output dim: 17, lower bound: -24.2953354, upper bound: 24.2962088
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 97.21
Output dim: 17, lower bound: -24.3322431, upper bound: 24.3322431

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -22.1175804, 15.9163780, -22.1258564, 15.8993616, -33.7311707, 33.6969604
1: -1.9297707, 26.0820045, -1.9244871, 26.0716476, -23.7357864, 23.7419701
2: 1.4778492, 24.3103867, 1.4973259, 24.2993965, -20.4559555, 20.4658031
3: -12.8998728, 14.6441011, -12.9169416, 14.6411324, -23.5933762, 23.6518917
4: -1.8369052, 26.2292233, -1.7912126, 26.2165108, -24.3335800, 24.3006783
5: 0.2835593, 23.3394756, 0.2708154, 23.3360901, -20.7138138, 20.7338448
6: -74.5362701, -34.0927353, -74.5228577, -34.1195831, -29.1895638, 29.2006683
7: -5.0097551, 19.2715912, -5.0067635, 19.2673759, -20.0796928, 20.0943546
8: -8.8625917, 20.5523300, -8.8390589, 20.5577621, -27.9720650, 27.9851685
9: -26.2215500, 10.9253216, -26.2473450, 10.9182615, -30.5836792, 30.6060944
10: -35.3576241, 10.9260330, -35.3748322, 10.9011192, -37.1029510, 37.0789642
11: -25.0106335, 9.9097805, -24.9980125, 9.8838806, -27.4995117, 27.4994049
12: -42.5483170, -6.0040283, -42.5531731, -6.0496421, -26.0738525, 26.0455055
13: -38.5257111, 20.1589394, -38.5337296, 20.1118164, -43.2709045, 43.3315201
14: 1.9104033, 49.0434227, 1.8974314, 49.0179558, -39.4572754, 39.4647217
15: -16.9428673, 19.4811764, -16.9060955, 19.4657688, -31.9091110, 31.8648987
16: -38.0568542, 6.7122822, -38.0731659, 6.7038002, -34.2440338, 34.2881126
17: 9.9435949, 55.0315399, 9.9265785, 54.9975471, -35.9670258, 35.9320602
18: -8.1514473, 38.7746277, -8.1024923, 38.7513962, -41.4535980, 41.3040390
19: -18.1418610, 17.3552132, -18.1352978, 17.3886414, -30.6824188, 30.6051941
20: -16.1487236, 18.6740704, -16.1328373, 18.6831989, -31.8515167, 31.8327179
21: -16.0617332, 19.2530975, -16.0521927, 19.2641678, -33.3896561, 33.3663940
22: -2.4093490, 31.4638519, -2.3976269, 31.4757366, -27.6083145, 27.5720749
23: -14.9020958, 15.2621527, -14.8978310, 15.2918577, -28.6034241, 28.5935059
24: -8.7990465, 29.2241001, -8.7822838, 29.2486134, -33.5206680, 33.4243736
25: 1.2013798, 38.9750290, 1.2090435, 38.9883423, -34.2363510, 34.1969452
26: -16.2496395, 24.2971439, -16.2352619, 24.3321323, -36.6505051, 36.5885124
27: -14.3342505, 21.6817169, -14.2999220, 21.7068691, -26.5736046, 26.6489258
28: -5.4946861, 23.8716469, -5.4892731, 23.8990841, -29.1564331, 29.1457596
29: -6.3095903, 24.1866817, -6.2973633, 24.1789322, -26.1129227, 26.1056538
30: -12.5355406, 22.0550232, -12.5181942, 22.0262470, -30.3674774, 30.4288712
31: -17.2293377, 27.2061310, -17.2205791, 27.2321014, -39.4156647, 39.2828445
32: -70.1799164, -28.9654388, -70.1707230, -28.9880486, -30.7479553, 30.7907753
33: -110.3192673, -44.8298569, -110.2957764, -44.8257065, -45.3350372, 45.2946548
34: -88.7446289, -42.1751099, -88.7221069, -42.1852455, -29.0230789, 29.0133514
35: -66.5573654, -20.9510231, -66.5428925, -20.9563007, -30.8226929, 30.8029251
36: -60.8192329, -16.1321602, -60.8119507, -16.1110039, -29.5137634, 29.4968872
37: -121.3775101, -55.2769318, -121.3489456, -55.2626686, -46.1752319, 46.0947189
38: -70.6724167, -15.6962109, -70.6584930, -15.6730309, -39.7459717, 39.6679382
39: -111.0492554, -44.4996758, -111.0329590, -44.4860382, -41.4675293, 41.4308243
40: -114.8150253, -64.8554535, -114.7874146, -64.8654861, -35.6740112, 35.5966034
41: -87.1660385, -42.6099358, -87.1501236, -42.6006317, -29.3850365, 29.4468803
42: -72.5506897, -36.0163994, -72.5414886, -36.0339050, -30.0320816, 30.0437927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1703

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1979827, upper bound: 24.3224800
time: 49.54 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1979827, upper bound: 24.3298664
time: 39.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.1392384, 15.9181623, -22.1694260, 15.9189100, -33.7738800, 33.7259026
1: -1.9448512, 26.0829678, -1.9547174, 26.0839310, -23.7633133, 23.7644882
2: 1.4618056, 24.3117905, 1.4652567, 24.3149776, -20.4871178, 20.4871788
3: -12.9032459, 14.6465464, -12.9213638, 14.6525354, -23.6130943, 23.6588669
4: -1.8718493, 26.2307701, -1.8610573, 26.2379780, -24.3898087, 24.3578873
5: 0.2809258, 23.3414555, 0.2657623, 23.3426323, -20.7439423, 20.7371750
6: -74.5375366, -34.0655479, -74.5419312, -34.0652542, -29.2352409, 29.2435341
7: -5.0116940, 19.2760105, -5.0133657, 19.2775230, -20.0947723, 20.1032257
8: -8.8809872, 20.5535927, -8.8759136, 20.5713234, -28.0092239, 27.9927177
9: -26.2234726, 10.9335546, -26.2523918, 10.9373369, -30.6082153, 30.6176872
10: -35.3591499, 10.9444485, -35.3892746, 10.9437885, -37.1349716, 37.1107674
11: -25.0120697, 9.9431000, -25.0170288, 9.9490280, -27.5572128, 27.5485916
12: -42.5509186, -5.9700823, -42.5843735, -5.9817166, -26.1211014, 26.1108398
13: -38.5266190, 20.1956978, -38.5558014, 20.1881104, -43.2955017, 43.4206009
14: 1.9014969, 49.0625610, 1.8550186, 49.0578537, -39.4829941, 39.5341949
15: -16.9802475, 19.4833679, -16.9802284, 19.4878311, -31.9637909, 31.9310684
16: -38.0596695, 6.7222314, -38.0783577, 6.7211380, -34.2878761, 34.2892532
17: 9.9370489, 55.0581017, 9.8841324, 55.0495110, -35.9941711, 35.9999771
18: -8.1904421, 38.7755051, -8.1854162, 38.7752876, -41.5131989, 41.3617020
19: -18.1455421, 17.3578625, -18.1452255, 17.3920174, -30.6880646, 30.6395493
20: -16.1500053, 18.6931000, -16.1511116, 18.7213821, -31.8856354, 31.8709030
21: -16.0649872, 19.2675686, -16.0675430, 19.2934399, -33.4233398, 33.4055099
22: -2.4179530, 31.4631310, -2.4166250, 31.4742661, -27.6162186, 27.5967102
23: -14.9057436, 15.2644434, -14.9078484, 15.2982903, -28.6095123, 28.6351967
24: -8.8147907, 29.2251282, -8.8159618, 29.2579231, -33.5509949, 33.4501610
25: 1.1968069, 38.9821472, 1.1941919, 39.0025253, -34.2520828, 34.2257996
26: -16.2556267, 24.2976665, -16.2547188, 24.3306675, -36.6538162, 36.6324387
27: -14.3590336, 21.6822720, -14.3519554, 21.7197266, -26.6191559, 26.6830082
28: -5.4976416, 23.8734608, -5.4990807, 23.9033470, -29.1682892, 29.1830063
29: -6.3120737, 24.1994648, -6.3156061, 24.2050896, -26.1283684, 26.1364899
30: -12.5379753, 22.0899715, -12.5411415, 22.0965004, -30.4198532, 30.4872055
31: -17.2378407, 27.2071743, -17.2412834, 27.2353973, -39.4210815, 39.3065796
32: -70.1819763, -28.9466248, -70.1863174, -28.9489784, -30.7799568, 30.8241844
33: -110.3418274, -44.8268356, -110.3418198, -44.7988739, -45.3928375, 45.3050308
34: -88.7623596, -42.1714668, -88.7589340, -42.1647873, -29.0597801, 29.0462646
35: -66.5716400, -20.9488201, -66.5735168, -20.9371910, -30.8537903, 30.8218384
36: -60.8220329, -16.1295185, -60.8247910, -16.1051960, -29.5202446, 29.5134315
37: -121.4039459, -55.2733459, -121.4062195, -55.2364388, -46.2398529, 46.0890884
38: -70.6814270, -15.6947947, -70.6836090, -15.6628962, -39.7618408, 39.6933441
39: -111.0671692, -44.4969444, -111.0683746, -44.4609909, -41.5041199, 41.4360504
40: -114.8369904, -64.8517685, -114.8337250, -64.8402634, -35.7478867, 35.5932732
41: -87.1804047, -42.6045380, -87.1800842, -42.5771751, -29.4341431, 29.4388447
42: -72.5523300, -35.9921341, -72.5554504, -35.9833755, -30.0830994, 30.0933609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1703

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2492263, upper bound: 24.2714578
time: 22.19 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2929621, upper bound: 24.3298662
time: 36.99 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -22.2092094, 15.9822979, -22.2047806, 15.9209919, -33.7637787, 33.8466034
1: -1.9719760, 26.1185417, -1.9714034, 26.0857010, -23.7817726, 23.8212776
2: 1.4586780, 24.3283463, 1.4577837, 24.3184090, -20.5014153, 20.5237751
3: -12.9353924, 14.7003536, -12.9361038, 14.6580811, -23.6463776, 23.6968060
4: -1.8688076, 26.2503700, -1.8757093, 26.2441292, -24.3747978, 24.4155045
5: 0.2499628, 23.3807945, 0.2534137, 23.3463440, -20.7618065, 20.8112259
6: -74.5768280, -34.0529366, -74.5459213, -34.0478668, -29.2918472, 29.2536316
7: -5.0162668, 19.2879734, -5.0144510, 19.2814751, -20.1075401, 20.1070232
8: -8.9202271, 20.5942345, -8.8843956, 20.5848827, -28.0710907, 28.0629807
9: -26.2740669, 11.0083055, -26.2734871, 10.9452991, -30.6585083, 30.7308083
10: -35.4139023, 11.0271606, -35.4108353, 10.9564152, -37.2008286, 37.2704048
11: -25.0565262, 9.9650526, -25.0213852, 9.9761963, -27.6341591, 27.5795631
12: -42.6100311, -5.9108958, -42.6089516, -5.9676433, -26.2101021, 26.2508354
13: -38.5785294, 20.2711754, -38.5777435, 20.2067833, -43.4291992, 43.4333382
14: 1.8063412, 49.1540833, 1.8178654, 49.0662766, -39.6067619, 39.6670761
15: -17.0015526, 19.5196495, -17.0047340, 19.4923668, -31.9454269, 32.0240860
16: -38.1015511, 6.7746525, -38.1016693, 6.7271585, -34.3133125, 34.3867912
17: 9.8377171, 55.1325989, 9.8430176, 55.0613251, -36.1253929, 36.1781998
18: -8.2073298, 38.7774658, -8.2072124, 38.7773514, -41.3726044, 41.4551620
19: -18.2405052, 17.4152470, -18.1516933, 17.4172096, -30.7959366, 30.6788063
20: -16.2303810, 18.7534866, -16.1527138, 18.7536411, -32.0116882, 31.9203644
21: -16.1416206, 19.3192043, -16.0719776, 19.3209343, -33.5340576, 33.4505768
22: -2.4621925, 31.4844742, -2.4223685, 31.4854946, -27.6653290, 27.6199150
23: -15.0008698, 15.3287592, -14.9123363, 15.3231497, -28.7880478, 28.6748810
24: -8.8913193, 29.2797775, -8.8271179, 29.2811642, -33.5835266, 33.5230713
25: 1.1373997, 39.0229568, 1.1880932, 39.0252686, -34.3303146, 34.2647018
26: -16.3404007, 24.3578453, -16.2621727, 24.3543663, -36.7743683, 36.6809464
27: -14.4651031, 21.7457523, -14.3635426, 21.7467690, -26.8635254, 26.7720413
28: -5.5794477, 23.9320297, -5.5024033, 23.9258118, -29.3168755, 29.2262115
29: -6.3468990, 24.2124062, -6.3202610, 24.2163010, -26.1832199, 26.1392574
30: -12.5791035, 22.1208534, -12.5451412, 22.1244793, -30.5263519, 30.4488831
31: -17.3143711, 27.2537918, -17.2519035, 27.2553654, -39.4292603, 39.3556976
32: -70.2011108, -28.9389400, -70.1903076, -28.9384212, -30.8228455, 30.8020077
33: -110.4148560, -44.7758255, -110.3486252, -44.7780151, -45.4282074, 45.3999176
34: -88.7695770, -42.1588783, -88.7663345, -42.1590424, -29.0638809, 29.0796471
35: -66.5953522, -20.9270592, -66.5799866, -20.9285412, -30.8682556, 30.8693314
36: -60.8906517, -16.0875587, -60.8285828, -16.0871487, -29.6235428, 29.5479202
37: -121.5092010, -55.2091713, -121.4225464, -55.2086525, -46.2562485, 46.2102051
38: -70.7656784, -15.6406431, -70.6910248, -15.6394949, -39.8180389, 39.7442856
39: -111.1617279, -44.4350471, -111.0737915, -44.4344254, -41.5919342, 41.5200195
40: -114.8777084, -64.8252335, -114.8450928, -64.8283615, -35.6232147, 35.7008247
41: -87.2565308, -42.5551796, -87.1877747, -42.5537567, -29.5746994, 29.5187492
42: -72.5966339, -35.9652786, -72.5586853, -35.9626770, -30.1733398, 30.1100349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=329, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1703

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2345382, upper bound: 24.3224799
time: 42.56 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3298661, upper bound: 24.3298663
time: 36.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 81.97 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.1979827, upper bound: 24.3224800
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.1979827, upper bound: 24.3298664
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.2492263, upper bound: 24.2714578
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.2929621, upper bound: 24.3298662
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.2345382, upper bound: 24.3224799
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 81.97
Output dim: 17, lower bound: -24.3298661, upper bound: 24.3298663

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -22.0774269, 15.9148188, -22.0387764, 15.8717804, -33.6580620, 33.6054420
1: -1.9256816, 26.0779114, -1.9141114, 26.0617409, -23.7186470, 23.7275887
2: 1.4839768, 24.3080139, 1.5131209, 24.2910271, -20.4379654, 20.4447117
3: -12.8988628, 14.6155882, -12.9115162, 14.5763388, -23.5223846, 23.6075268
4: -1.8153398, 26.2275467, -1.7413790, 26.2109890, -24.2994881, 24.2420502
5: 0.2852054, 23.3279610, 0.2833762, 23.3101540, -20.6850471, 20.7038193
6: -74.5333862, -34.1397972, -74.4811249, -34.2199783, -29.0911713, 29.1246643
7: -5.0081720, 19.2530880, -4.9847903, 19.2265625, -20.0414352, 20.0540295
8: -8.8541040, 20.5494995, -8.8176517, 20.5498161, -27.9431381, 27.9494667
9: -26.2112923, 10.9219265, -26.2230835, 10.9152040, -30.5674057, 30.5797386
10: -35.3427124, 10.9199123, -35.3410950, 10.8883276, -37.0653992, 37.0246201
11: -25.0075283, 9.8959332, -24.9840069, 9.8535433, -27.4694138, 27.4765701
12: -42.5450668, -6.0585723, -42.5144196, -6.1668515, -25.9485550, 25.9385033
13: -38.5242729, 20.1025600, -38.5141182, 19.9888954, -43.1497345, 43.2582664
14: 1.9230204, 49.0108109, 1.9600353, 48.9476395, -39.3791504, 39.3785706
15: -16.8956623, 19.4789448, -16.8065758, 19.4409370, -31.8360519, 31.7602234
16: -38.0258408, 6.7110195, -38.0038643, 6.7022076, -34.2085342, 34.2148438
17: 9.9502220, 54.9687119, 9.9744501, 54.8656769, -35.8262405, 35.8253059
18: -8.0994291, 38.7738647, -7.9869556, 38.7337112, -41.3828049, 41.1919098
19: -18.0969353, 17.3548527, -18.0371208, 17.3817291, -30.6063538, 30.4878693
20: -16.1470127, 18.6578217, -16.1108627, 18.6475525, -31.8064194, 31.7944756
21: -16.0419693, 19.2513943, -16.0055580, 19.2628307, -33.3554764, 33.3046684
22: -2.3957205, 31.4630280, -2.3666067, 31.4645309, -27.5927124, 27.5443192
23: -14.8666697, 15.2611952, -14.8215199, 15.2590694, -28.5513229, 28.5161858
24: -8.7563934, 29.2232609, -8.6906281, 29.2244282, -33.4425812, 33.3183022
25: 1.2336311, 38.9727478, 1.2776432, 38.9505081, -34.1754684, 34.1391068
26: -16.2306232, 24.2954483, -16.1824417, 24.3282776, -36.6131821, 36.5178680
27: -14.3005695, 21.6810284, -14.2222729, 21.7025185, -26.5345001, 26.5637321
28: -5.4771137, 23.8698769, -5.4497957, 23.8795242, -29.1236343, 29.0982323
29: -6.3028917, 24.1838646, -6.2796116, 24.1688862, -26.0957947, 26.0838108
30: -12.5322247, 22.0107021, -12.4892063, 21.9321423, -30.2705078, 30.3626556
31: -17.1728096, 27.2059135, -17.1003647, 27.2034740, -39.3204193, 39.1569061
32: -70.1772614, -29.0077400, -70.1389313, -29.0797691, -30.6477814, 30.7096424
33: -110.2893066, -44.8337746, -110.2330475, -44.8673325, -45.2702103, 45.2273407
34: -88.7371750, -42.1885376, -88.7037506, -42.2277603, -28.9519577, 28.9596100
35: -66.5436401, -20.9545898, -66.5135422, -20.9772186, -30.7854538, 30.7674713
36: -60.8161125, -16.1386013, -60.8017769, -16.1257305, -29.4937897, 29.4726486
37: -121.3118896, -55.2785568, -121.2073517, -55.3187180, -46.0487823, 45.9544525
38: -70.6662140, -15.7009163, -70.6443787, -15.6837940, -39.7300797, 39.6499023
39: -111.0096436, -44.5012512, -110.9489899, -44.5241699, -41.3937836, 41.3504791
40: -114.7891464, -64.8566971, -114.7299194, -64.8895264, -35.6247253, 35.5345421
41: -87.1462708, -42.6138535, -87.1067047, -42.6236458, -29.3590126, 29.4108429
42: -72.5492706, -36.0361786, -72.5374298, -36.0815506, -29.9718018, 30.0052071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1528904, upper bound: 24.3200856
time: 53.82 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1528904, upper bound: 24.3201139
time: 43.47 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -22.1161156, 15.9161568, -22.1218529, 15.8987303, -33.7300568, 33.6517029
1: -1.9294879, 26.0806313, -1.9237301, 26.0680485, -23.7395935, 23.7375679
2: 1.4781649, 24.3099174, 1.4981041, 24.2981415, -20.4551163, 20.4634876
3: -12.8997746, 14.6405659, -12.9166508, 14.6314631, -23.5611610, 23.6485081
4: -1.8348894, 26.2291069, -1.7859497, 26.2162151, -24.3336830, 24.2965088
5: 0.2838731, 23.3378658, 0.2716222, 23.3316879, -20.6896896, 20.7310543
6: -74.5360413, -34.0943146, -74.5222168, -34.1239090, -29.1237144, 29.1980972
7: -5.0094786, 19.2692719, -5.0060067, 19.2610798, -20.0440483, 20.0900536
8: -8.8613291, 20.5521450, -8.8356628, 20.5573120, -27.9658737, 27.9828758
9: -26.2192764, 10.9250031, -26.2411137, 10.9174919, -30.5806007, 30.5956116
10: -35.3541946, 10.9257097, -35.3653526, 10.9002762, -37.0965271, 37.0607681
11: -25.0103416, 9.9089832, -24.9972305, 9.8818197, -27.4913025, 27.4930916
12: -42.5481567, -6.0062242, -42.5527649, -6.0556674, -25.9896011, 26.0443954
13: -38.5254822, 20.1574783, -38.5330849, 20.1077042, -43.2158813, 43.3290558
14: 1.9118824, 49.0425720, 1.9014292, 49.0160217, -39.4042664, 39.4601898
15: -16.9421482, 19.4809723, -16.9047604, 19.4652443, -31.9064560, 31.8147430
16: -38.0519829, 6.7121406, -38.0598450, 6.7033215, -34.2398186, 34.2547684
17: 9.9443083, 55.0305786, 9.9285269, 54.9950104, -35.9062500, 35.9300728
18: -8.1490269, 38.7745667, -8.0962515, 38.7512817, -41.4504700, 41.2851639
19: -18.1362267, 17.3551979, -18.1198902, 17.3886070, -30.6802368, 30.5382538
20: -16.1485672, 18.6734943, -16.1324387, 18.6818810, -31.8304291, 31.8302307
21: -16.0587521, 19.2530613, -16.0440216, 19.2639370, -33.3846741, 33.3566856
22: -2.4085975, 31.4637985, -2.3955564, 31.4756470, -27.6051483, 27.5631981
23: -14.9008484, 15.2620201, -14.8943977, 15.2915125, -28.5958176, 28.5605965
24: -8.7969437, 29.2240543, -8.7765684, 29.2484512, -33.5185928, 33.3707619
25: 1.2023392, 38.9749298, 1.2116270, 38.9881516, -34.2333832, 34.1433411
26: -16.2466526, 24.2969589, -16.2272072, 24.3316879, -36.6494751, 36.5811501
27: -14.3320465, 21.6816673, -14.2939348, 21.7066193, -26.5705185, 26.6342564
28: -5.4937210, 23.8714142, -5.4866076, 23.8984451, -29.1500702, 29.1303558
29: -6.3082705, 24.1865501, -6.2938538, 24.1785450, -26.1110725, 26.0973949
30: -12.5350800, 22.0547447, -12.5168724, 22.0254650, -30.3243637, 30.4272690
31: -17.2254200, 27.2061424, -17.2100639, 27.2320652, -39.4137268, 39.1997452
32: -70.1797409, -28.9686279, -70.1701965, -28.9962921, -30.6937408, 30.7881641
33: -110.3180695, -44.8301392, -110.2929230, -44.8265076, -45.3278809, 45.2511444
34: -88.7431717, -42.1776962, -88.7181549, -42.1922569, -29.0140228, 29.0186844
35: -66.5562363, -20.9515076, -66.5402145, -20.9576397, -30.8180237, 30.7768974
36: -60.8190460, -16.1338501, -60.8114700, -16.1155434, -29.5003700, 29.4951935
37: -121.3751221, -55.2770958, -121.3424683, -55.2632751, -46.1723633, 45.9813080
38: -70.6721268, -15.6975393, -70.6577835, -15.6766272, -39.7465897, 39.6642075
39: -111.0474243, -44.4998474, -111.0281525, -44.4865189, -41.4649658, 41.3589172
40: -114.8144836, -64.8556595, -114.7859726, -64.8661652, -35.6722946, 35.5668411
41: -87.1653976, -42.6106262, -87.1484680, -42.6026306, -29.3804092, 29.4326363
42: -72.5503845, -36.0187225, -72.5406647, -36.0401878, -30.0245743, 30.0489006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2118056, upper bound: 24.3274708
time: 40.27 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2545309, upper bound: 24.3274924
time: 45.05 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.1353035, 15.9175282, -22.1679420, 15.9186792, -33.7285881, 33.7248116
1: -1.9440694, 26.0793667, -1.9544330, 26.0825920, -23.7589264, 23.7682953
2: 1.4626353, 24.3105354, 1.4655633, 24.3145123, -20.4848175, 20.4863510
3: -12.9029541, 14.6368923, -12.9212475, 14.6489944, -23.6096802, 23.6266155
4: -1.8666124, 26.2304459, -1.8590291, 26.2378826, -24.3856049, 24.3579788
5: 0.2817531, 23.3370667, 0.2660680, 23.3410320, -20.7411346, 20.7130203
6: -74.5369263, -34.0698700, -74.5417480, -34.0668373, -29.2326431, 29.1776581
7: -5.0109587, 19.2697296, -5.0131063, 19.2751999, -20.0904770, 20.0676117
8: -8.8775606, 20.5531082, -8.8746233, 20.5711231, -28.0068741, 27.9865532
9: -26.2172623, 10.9327364, -26.2500916, 10.9370317, -30.5977249, 30.6146088
10: -35.3497162, 10.9435282, -35.3858490, 10.9434605, -37.1167679, 37.1042862
11: -25.0112629, 9.9410686, -25.0167561, 9.9482327, -27.5509186, 27.5403595
12: -42.5504456, -5.9761147, -42.5842056, -5.9839263, -26.1199837, 26.0265808
13: -38.5259323, 20.1916294, -38.5555573, 20.1866035, -43.2930603, 43.3654976
14: 1.9054413, 49.0605774, 1.8564806, 49.0569763, -39.4784126, 39.4811134
15: -16.9788933, 19.4828186, -16.9794693, 19.4876556, -31.9135818, 31.9284592
16: -38.0463943, 6.7217269, -38.0735321, 6.7209644, -34.2544899, 34.2850723
17: 9.9390078, 55.0555344, 9.8848372, 55.0485764, -35.9921722, 35.9391747
18: -8.1843176, 38.7753716, -8.1829729, 38.7752266, -41.4942780, 41.3585510
19: -18.1300964, 17.3578033, -18.1396027, 17.3920040, -30.6211090, 30.6373634
20: -16.1496010, 18.6917820, -16.1509399, 18.7207985, -31.8830566, 31.8498001
21: -16.0568047, 19.2673683, -16.0645733, 19.2933674, -33.4135895, 33.4005508
22: -2.4158716, 31.4630318, -2.4158916, 31.4742184, -27.6073074, 27.5935402
23: -14.9022818, 15.2641087, -14.9065666, 15.2981701, -28.5764923, 28.6275406
24: -8.8090534, 29.2249641, -8.8138647, 29.2578621, -33.4973755, 33.4480858
25: 1.1994305, 38.9819717, 1.1951313, 39.0024452, -34.1984558, 34.2228165
26: -16.2475300, 24.2972527, -16.2517719, 24.3305283, -36.6464005, 36.6314011
27: -14.3529873, 21.6820297, -14.3497581, 21.7196465, -26.6044655, 26.6799507
28: -5.4949589, 23.8728371, -5.4980831, 23.9031353, -29.1526985, 29.1766052
29: -6.3085289, 24.1990929, -6.3142786, 24.2049446, -26.1201324, 26.1346226
30: -12.5366669, 22.0892067, -12.5406656, 22.0962143, -30.4182701, 30.4440804
31: -17.2272911, 27.2071533, -17.2374153, 27.2353840, -39.3379364, 39.3046570
32: -70.1815033, -28.9548512, -70.1861267, -28.9521255, -30.7773209, 30.7699394
33: -110.3390656, -44.8275528, -110.3406906, -44.7991104, -45.3494568, 45.2977982
34: -88.7583923, -42.1786156, -88.7575226, -42.1673737, -29.0650177, 29.0371590
35: -66.5689392, -20.9501896, -66.5723572, -20.9376469, -30.8277817, 30.8170853
36: -60.8215408, -16.1340485, -60.8245926, -16.1068516, -29.5185585, 29.5000229
37: -121.3974152, -55.2739868, -121.4038391, -55.2366905, -46.1263580, 46.0861435
38: -70.6807861, -15.6983461, -70.6833649, -15.6642551, -39.7580872, 39.6939621
39: -111.0623398, -44.4974594, -111.0666046, -44.4611053, -41.4322662, 41.4334793
40: -114.8355179, -64.8524551, -114.8331757, -64.8404999, -35.7181549, 35.5915527
41: -87.1787186, -42.6065674, -87.1794586, -42.5778656, -29.4198723, 29.4342232
42: -72.5514908, -35.9985085, -72.5551376, -35.9857254, -30.0882416, 30.0858192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1693

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2041305, upper bound: 24.3274706
time: 47.28 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2905861, upper bound: 24.3274922
time: 60.76 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -22.1690598, 15.9806929, -22.1177197, 15.8934479, -33.6906509, 33.7549973
1: -1.9678671, 26.1144485, -1.9609606, 26.0757484, -23.7646255, 23.8068733
2: 1.4648087, 24.3260059, 1.4736362, 24.3100433, -20.4834099, 20.5025826
3: -12.9343891, 14.6718407, -12.9306946, 14.5933952, -23.5754013, 23.6524429
4: -1.8472507, 26.2486782, -1.8258109, 26.2385712, -24.3407211, 24.3568840
5: 0.2516003, 23.3692875, 0.2660131, 23.3204308, -20.7329903, 20.7812119
6: -74.5739136, -34.1000366, -74.5042114, -34.1483345, -29.1933212, 29.1776047
7: -5.0146661, 19.2694759, -4.9924688, 19.2406654, -20.0693054, 20.0667458
8: -8.9117165, 20.5914040, -8.8628635, 20.5769100, -28.0421906, 28.0270844
9: -26.2637920, 11.0049152, -26.2492409, 10.9422331, -30.6422043, 30.7043762
10: -35.3989143, 11.0210953, -35.3770790, 10.9435215, -37.1632233, 37.2161331
11: -25.0533619, 9.9512119, -25.0073948, 9.9458122, -27.6040344, 27.5567169
12: -42.6067505, -5.9654574, -42.5702705, -6.0848269, -26.0848083, 26.1438675
13: -38.5770149, 20.2148037, -38.5580444, 20.0838833, -43.3080368, 43.3600769
14: 1.8189592, 49.1215134, 1.8804588, 48.9959335, -39.5286331, 39.5808411
15: -16.9542999, 19.5174179, -16.9051437, 19.4675369, -31.8723679, 31.9194145
16: -38.0705872, 6.7733536, -38.0323334, 6.7255807, -34.2777557, 34.3135605
17: 9.8443174, 55.0697708, 9.8909273, 54.9293861, -35.9845581, 36.0714340
18: -8.1553421, 38.7767029, -8.0916843, 38.7596893, -41.3018799, 41.3428192
19: -18.1956444, 17.4148941, -18.0535336, 17.4102993, -30.7199707, 30.5615425
20: -16.2286816, 18.7372150, -16.1307487, 18.7179794, -31.9664841, 31.8820038
21: -16.1218681, 19.3174210, -16.0253468, 19.3196030, -33.4998932, 33.3887177
22: -2.4485893, 31.4836712, -2.3913169, 31.4742355, -27.6497307, 27.5920563
23: -14.9654293, 15.3278027, -14.8360147, 15.2903681, -28.7360229, 28.5972862
24: -8.8486814, 29.2789307, -8.7354488, 29.2569675, -33.5054932, 33.4170074
25: 1.1696548, 39.0205421, 1.2566700, 38.9872360, -34.2694702, 34.2068329
26: -16.3216171, 24.3561440, -16.2093163, 24.3505173, -36.7369080, 36.6102142
27: -14.4314661, 21.7450371, -14.2858305, 21.7424164, -26.8243637, 26.6866360
28: -5.5618792, 23.9302597, -5.4628906, 23.9062462, -29.2840195, 29.1784248
29: -6.3401270, 24.2095947, -6.3024759, 24.2062664, -26.1661110, 26.1174259
30: -12.5757685, 22.0765324, -12.5162086, 22.0303555, -30.4293671, 30.3826675
31: -17.2579231, 27.2535839, -17.1315918, 27.2267246, -39.3340759, 39.2297287
32: -70.1984558, -28.9812088, -70.1585236, -29.0301743, -30.7225723, 30.7207928
33: -110.3849335, -44.7796021, -110.2858810, -44.8196411, -45.3637543, 45.3326035
34: -88.7621231, -42.1722908, -88.7479935, -42.2017288, -28.9927444, 29.0258675
35: -66.5815811, -20.9306393, -66.5505524, -20.9495296, -30.8310471, 30.8338776
36: -60.8875732, -16.0939903, -60.8183670, -16.1018867, -29.6035538, 29.5236855
37: -121.4435730, -55.2107964, -121.2809601, -55.2647972, -46.1297379, 46.0697250
38: -70.7595367, -15.6453552, -70.6768036, -15.6502838, -39.8021393, 39.7262344
39: -111.1221161, -44.4365578, -110.9897537, -44.4725761, -41.5182953, 41.4396439
40: -114.8517838, -64.8264389, -114.7875671, -64.8523178, -35.5739288, 35.6387138
41: -87.2367706, -42.5591431, -87.1443939, -42.5767899, -29.5487251, 29.4825287
42: -72.5952301, -35.9851189, -72.5546036, -36.0105324, -30.1130905, 30.0714531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2321629, upper bound: 24.2773684
time: 51.88 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2690828, upper bound: 24.3201137
time: 46.43 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -22.2077217, 15.9820566, -22.2008133, 15.9203939, -33.7626724, 33.8013000
1: -1.9717169, 26.1171970, -1.9706278, 26.0820847, -23.7856064, 23.8169022
2: 1.4589810, 24.3279095, 1.4585969, 24.3171310, -20.5006027, 20.5214615
3: -12.9352789, 14.6968060, -12.9358244, 14.6484241, -23.6141281, 23.6934166
4: -1.8667727, 26.2502403, -1.8704624, 26.2437935, -24.3749084, 24.4113197
5: 0.2502489, 23.3791943, 0.2542782, 23.3419552, -20.7376404, 20.8084602
6: -74.5765610, -34.0545044, -74.5453186, -34.0522118, -29.2259064, 29.2510300
7: -5.0159941, 19.2856560, -5.0137072, 19.2751999, -20.0719223, 20.1027012
8: -8.9189539, 20.5940552, -8.8809528, 20.5844002, -28.0649109, 28.0606308
9: -26.2717934, 11.0080080, -26.2672672, 10.9445019, -30.6554222, 30.7202988
10: -35.4104424, 11.0268660, -35.4013977, 10.9555092, -37.1943359, 37.2521973
11: -25.0562344, 9.9642429, -25.0205860, 9.9741325, -27.6259308, 27.5732536
12: -42.6098633, -5.9130673, -42.6084976, -5.9736772, -26.1258392, 26.2497177
13: -38.5782852, 20.2696705, -38.5770226, 20.2027473, -43.3741226, 43.4308929
14: 1.8077841, 49.1532440, 1.8218985, 49.0643234, -39.5537186, 39.6625214
15: -17.0007725, 19.5194550, -17.0033894, 19.4918633, -31.9428101, 31.9739037
16: -38.0966492, 6.7744517, -38.0883484, 6.7266855, -34.3090973, 34.3533974
17: 9.8384571, 55.1316299, 9.8450127, 55.0587425, -36.0646439, 36.1761932
18: -8.2048969, 38.7774048, -8.2009811, 38.7772293, -41.3694458, 41.4362640
19: -18.2348862, 17.4152412, -18.1362801, 17.4171753, -30.7937393, 30.6118736
20: -16.2302151, 18.7529030, -16.1523113, 18.7523117, -31.9905701, 31.9177780
21: -16.1386433, 19.3190994, -16.0637703, 19.3207569, -33.5290985, 33.4408379
22: -2.4614391, 31.4844112, -2.4203119, 31.4853668, -27.6621513, 27.6109886
23: -14.9996109, 15.3286295, -14.9088936, 15.3227921, -28.7804260, 28.6418266
24: -8.8892097, 29.2797394, -8.8213778, 29.2809944, -33.5815125, 33.4694405
25: 1.1383400, 39.0228806, 1.1906595, 39.0250664, -34.3273315, 34.2111511
26: -16.3374596, 24.3576889, -16.2540779, 24.3539124, -36.7733383, 36.6735382
27: -14.4629469, 21.7456665, -14.3575478, 21.7465363, -26.8604507, 26.7573662
28: -5.5785017, 23.9318161, -5.4997044, 23.9251785, -29.3104706, 29.2105980
29: -6.3455496, 24.2122612, -6.3167143, 24.2158813, -26.1813469, 26.1310215
30: -12.5786381, 22.1205482, -12.5438337, 22.1237011, -30.4832497, 30.4472542
31: -17.3104992, 27.2537937, -17.2413044, 27.2553291, -39.4273376, 39.2725449
32: -70.2009277, -28.9420776, -70.1897888, -28.9466972, -30.7685890, 30.7993679
33: -110.4136658, -44.7761383, -110.3457336, -44.7787323, -45.4210358, 45.3564911
34: -88.7681427, -42.1614494, -88.7623749, -42.1661758, -29.0547791, 29.0848999
35: -66.5941467, -20.9275284, -66.5772934, -20.9298973, -30.8635101, 30.8433685
36: -60.8904610, -16.0892143, -60.8281403, -16.0917225, -29.6101761, 29.5462189
37: -121.5068054, -55.2094116, -121.4160385, -55.2093124, -46.2533112, 46.0968018
38: -70.7654190, -15.6419973, -70.6902771, -15.6431265, -39.8186340, 39.7405396
39: -111.1599426, -44.4352188, -111.0688629, -44.4349098, -41.5893402, 41.4481583
40: -114.8771591, -64.8254547, -114.8436356, -64.8290024, -35.6215210, 35.6711082
41: -87.2559204, -42.5559196, -87.1861343, -42.5557518, -29.5700684, 29.5044460
42: -72.5963593, -35.9676094, -72.5578308, -35.9690247, -30.1658249, 30.1151733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1693

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2321629, upper bound: 24.2847449
time: 40.81 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3274921, upper bound: 24.3274922
time: 23.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 66.97 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.1528904, upper bound: 24.3200856
IS_A1_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.1528904, upper bound: 24.3201139
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2118056, upper bound: 24.3274708
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2545309, upper bound: 24.3274924
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2041305, upper bound: 24.3274706
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2905861, upper bound: 24.3274922
IS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2321629, upper bound: 24.2773684
IS_A2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2690828, upper bound: 24.3201137
IS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.2321629, upper bound: 24.2847449
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 66.97
Output dim: 17, lower bound: -24.3274921, upper bound: 24.3274922

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -22.1003647, 15.8636703, -22.1152534, 15.8767853, -33.6905556, 33.5899429
1: -1.9248009, 26.0249367, -1.9217684, 26.0447407, -23.7112961, 23.6795807
2: 1.4822776, 24.2605896, 1.4998507, 24.2775097, -20.4280624, 20.4090042
3: -12.8938074, 14.5981932, -12.9141560, 14.6136494, -23.5344086, 23.5985985
4: -1.8281553, 26.1718845, -1.7830946, 26.1922283, -24.3018303, 24.2342072
5: 0.2887473, 23.3068810, 0.2736659, 23.3187065, -20.6685448, 20.6932831
6: -74.4654236, -34.0962906, -74.4925995, -34.1247215, -29.0641403, 29.1706924
7: -5.0034618, 19.2339420, -5.0035000, 19.2462883, -20.0218353, 20.0495281
8: -8.8551569, 20.4929466, -8.8330574, 20.5325699, -27.9289017, 27.9148941
9: -26.2140446, 10.8649817, -26.2389259, 10.8920507, -30.5441666, 30.5226822
10: -35.3433838, 10.8840694, -35.3607750, 10.8828325, -37.0670090, 37.0052643
11: -24.9840584, 9.9047794, -24.9862213, 9.8800764, -27.4512634, 27.4716873
12: -42.4582710, -6.0101542, -42.5151596, -6.0573273, -25.8956985, 26.0013809
13: -38.4468765, 20.1399403, -38.5002785, 20.1003609, -43.1307602, 43.2749786
14: 1.9419737, 49.0052490, 1.9140272, 48.9999733, -39.3509903, 39.4084549
15: -16.9371433, 19.4263458, -16.9026566, 19.4422531, -31.8761902, 31.7525520
16: -38.0437279, 6.6455517, -38.0564156, 6.6753483, -34.2019730, 34.1817474
17: 10.0646286, 55.0272980, 9.9788694, 54.9936371, -35.7766647, 35.8732033
18: -8.1322460, 38.7029800, -8.0892210, 38.7213097, -41.3996735, 41.2042007
19: -18.1063271, 17.3527298, -18.1073742, 17.3875732, -30.6317368, 30.5153008
20: -16.1076012, 18.6651344, -16.1152191, 18.6783485, -31.7773590, 31.8003922
21: -16.0311623, 19.2466469, -16.0324688, 19.2612572, -33.3274231, 33.3261261
22: -2.3572798, 31.4579487, -2.3738089, 31.4731617, -27.5486908, 27.5352364
23: -14.8788261, 15.2558260, -14.8851080, 15.2889166, -28.5435867, 28.5317841
24: -8.7754288, 29.2168789, -8.7675152, 29.2454453, -33.4778214, 33.3471985
25: 1.2349987, 38.9641609, 1.2252846, 38.9836540, -34.1672668, 34.1070442
26: -16.2207432, 24.2829170, -16.2163353, 24.3257008, -36.6035919, 36.5493011
27: -14.3173685, 21.6345291, -14.2877464, 21.6869774, -26.5340347, 26.5789757
28: -5.4681740, 23.8627052, -5.4759026, 23.8948021, -29.0999146, 29.1015205
29: -6.2506351, 24.1811180, -6.2697153, 24.1762428, -26.0480995, 26.0667439
30: -12.4760647, 22.0466805, -12.4921436, 22.0220947, -30.2613449, 30.3946075
31: -17.1980362, 27.2013607, -17.1985741, 27.2301140, -39.3608780, 39.1716309
32: -70.1085663, -28.9762840, -70.1399384, -28.9995270, -30.6166763, 30.7469654
33: -110.2090759, -44.8388634, -110.2474289, -44.8300056, -45.2149353, 45.1959991
34: -88.6763153, -42.1820526, -88.6899414, -42.1940880, -28.9448433, 28.9853249
35: -66.4842987, -20.9555836, -66.5101852, -20.9593525, -30.7424622, 30.7413521
36: -60.7572098, -16.1377907, -60.7846527, -16.1172466, -29.4309540, 29.4616776
37: -121.2700043, -55.2815018, -121.2984085, -55.2651024, -46.0540466, 45.9258347
38: -70.6075821, -15.7027931, -70.6304550, -15.6788073, -39.6700897, 39.6268158
39: -110.9531326, -44.5033875, -110.9886475, -44.4879303, -41.3673019, 41.3146133
40: -114.7798691, -64.8591309, -114.7713776, -64.8676376, -35.6333160, 35.5330887
41: -87.1186371, -42.6160812, -87.1287613, -42.6049309, -29.3361053, 29.4066391
42: -72.5129013, -36.0243225, -72.5245819, -36.0425262, -29.9858475, 30.0266037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2070735, upper bound: 24.2521524
time: 41.46 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2101070, upper bound: 24.3257100
time: 42.99 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -22.1956444, 15.9222507, -22.1189919, 15.8937960, -33.8044128, 33.6523285
1: -2.0341656, 26.0812836, -1.9228656, 26.0630798, -23.8399353, 23.7330513
2: 1.4186304, 24.3135796, 1.4989455, 24.2934933, -20.5094604, 20.4627800
3: -12.9281721, 14.6446180, -12.9156151, 14.6260118, -23.5846481, 23.6475945
4: -1.9099102, 26.2347260, -1.7846394, 26.2110500, -24.4070396, 24.2965317
5: 0.2383876, 23.3438416, 0.2725430, 23.3284454, -20.7339745, 20.7308445
6: -74.5490265, -34.0205688, -74.5133514, -34.1245613, -29.1411514, 29.2533455
7: -5.0630922, 19.2746563, -5.0050058, 19.2576160, -20.0947495, 20.0922050
8: -8.9933558, 20.5494270, -8.8342953, 20.5518112, -28.0965500, 27.9703445
9: -26.2867928, 10.9381046, -26.2397308, 10.9116764, -30.6480141, 30.5959663
10: -35.3821220, 10.9431362, -35.3628845, 10.8927689, -37.1529770, 37.0683250
11: -25.0169544, 9.9394894, -24.9892960, 9.8809128, -27.4902649, 27.5455894
12: -42.5476685, -5.9254694, -42.5449142, -6.0564303, -25.9740601, 26.1218796
13: -38.5299950, 20.3044243, -38.5267410, 20.1043892, -43.2055511, 43.4749641
14: 1.8159637, 49.0501595, 1.9054155, 49.0123215, -39.4855194, 39.4598427
15: -17.0376568, 19.4976845, -16.9038124, 19.4598656, -32.0151749, 31.8196106
16: -38.1811943, 6.7135115, -38.0578461, 6.6942081, -34.3729782, 34.2451859
17: 9.9136257, 55.1226501, 9.9387684, 54.9939537, -35.9198456, 36.0146904
18: -8.2336102, 38.7733955, -8.0934639, 38.7450371, -41.5293579, 41.2750168
19: -18.1456890, 17.4219093, -18.1138153, 17.3883553, -30.6670456, 30.6267090
20: -16.1568165, 18.7428074, -16.1282692, 18.6806202, -31.8264160, 31.9090042
21: -16.0839920, 19.3160000, -16.0397720, 19.2632790, -33.3673401, 33.4577866
22: -2.4188833, 31.5521336, -2.3905249, 31.4748650, -27.6108704, 27.6525612
23: -14.9120893, 15.3009167, -14.8903942, 15.2903242, -28.5673065, 28.6425552
24: -8.8086033, 29.2550354, -8.7700605, 29.2472439, -33.5081329, 33.4358635
25: 1.1827698, 39.0376740, 1.2196836, 38.9863968, -34.2056122, 34.2644730
26: -16.2860031, 24.3094311, -16.2234268, 24.3278961, -36.6522980, 36.6324539
27: -14.4264736, 21.6848564, -14.2913857, 21.7026405, -26.6678085, 26.6276932
28: -5.5052528, 23.9187279, -5.4822931, 23.8971710, -29.1333771, 29.2147255
29: -6.3206339, 24.2558556, -6.2881441, 24.1777420, -26.1184883, 26.1629963
30: -12.5605345, 22.1554337, -12.5110807, 22.0242748, -30.3421669, 30.5247269
31: -17.2499123, 27.2402840, -17.2047348, 27.2312431, -39.3974457, 39.2848206
32: -70.1792679, -28.8708000, -70.1605148, -28.9976387, -30.6875076, 30.8777008
33: -110.3185196, -44.6517296, -110.2835617, -44.8280334, -45.3027420, 45.4441299
34: -88.7466431, -42.0482330, -88.7108917, -42.1933250, -29.0063553, 29.1421356
35: -66.5571289, -20.8351212, -66.5335846, -20.9588470, -30.8044968, 30.8987617
36: -60.8219681, -16.0321941, -60.8056107, -16.1164742, -29.4897308, 29.5937004
37: -121.3822937, -55.1925774, -121.3323517, -55.2643661, -46.1562195, 46.0730057
38: -70.6796570, -15.6377249, -70.6491394, -15.6782961, -39.7392426, 39.7261429
39: -111.0487213, -44.3604050, -111.0195694, -44.4871559, -41.4488754, 41.5052719
40: -114.8354187, -64.8297882, -114.7789764, -64.8674927, -35.7255402, 35.5555038
41: -87.1794815, -42.5441322, -87.1405640, -42.6037750, -29.4063721, 29.4847221
42: -72.5772095, -35.9640579, -72.5358505, -36.0414276, -30.0466232, 30.0996475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1500930, upper bound: 24.2521764
time: 49.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2528311, upper bound: 24.3257339
time: 39.23 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -22.1195812, 15.8650370, -22.1613770, 15.8967190, -33.6890488, 33.6630478
1: -1.9394000, 26.0236645, -1.9524937, 26.0592766, -23.7306137, 23.7102661
2: 1.4667392, 24.2612000, 1.4673047, 24.2938709, -20.4577713, 20.4318352
3: -12.8970032, 14.5944824, -12.9187565, 14.6311760, -23.5829544, 23.5767078
4: -1.8598351, 26.1732349, -1.8561676, 26.2138977, -24.3537827, 24.2956734
5: 0.2866106, 23.3060684, 0.2681122, 23.3280830, -20.7200050, 20.6752243
6: -74.4663086, -34.0718460, -74.5121002, -34.0676842, -29.1730881, 29.1502495
7: -5.0049500, 19.2343922, -5.0105524, 19.2603722, -20.0682831, 20.0270538
8: -8.8713722, 20.4939423, -8.8720503, 20.5463676, -27.9698563, 27.9185982
9: -26.2120342, 10.8727732, -26.2479057, 10.9116325, -30.5612869, 30.5417290
10: -35.3389168, 10.9019098, -35.3813210, 10.9260321, -37.0872040, 37.0487671
11: -24.9850101, 9.9368534, -25.0057049, 9.9464855, -27.5108910, 27.5189209
12: -42.4605789, -5.9800701, -42.5465927, -5.9855881, -26.0260849, 25.9835587
13: -38.4474411, 20.1741333, -38.5226822, 20.1792488, -43.2079086, 43.3114090
14: 1.9355373, 49.0232544, 1.8691025, 49.0409279, -39.4251518, 39.4293289
15: -16.9738464, 19.4281654, -16.9773636, 19.4646797, -31.8833008, 31.8663597
16: -38.0381317, 6.6550574, -38.0700836, 6.6929970, -34.2166405, 34.2120819
17: 10.0593138, 55.0522614, 9.9352064, 55.0472145, -35.8625870, 35.8823280
18: -8.1675425, 38.7037430, -8.1759434, 38.7452774, -41.4435425, 41.2775497
19: -18.1002197, 17.3553638, -18.1270409, 17.3909531, -30.5726242, 30.6144180
20: -16.1086121, 18.6834297, -16.1337433, 18.7172794, -31.8299866, 31.8199806
21: -16.0291977, 19.2609653, -16.0530281, 19.2906837, -33.3563004, 33.3700333
22: -2.3645906, 31.4572277, -2.3941317, 31.4717331, -27.5508575, 27.5655975
23: -14.8802757, 15.2579279, -14.8973179, 15.2955379, -28.5242920, 28.5987740
24: -8.7875767, 29.2177887, -8.8048353, 29.2548695, -33.4565811, 33.4245567
25: 1.2320585, 38.9712105, 1.2088280, 38.9979477, -34.1323547, 34.1865005
26: -16.2216187, 24.2831955, -16.2409077, 24.3245392, -36.6005096, 36.5995483
27: -14.3383083, 21.6349144, -14.3436260, 21.6999340, -26.5679855, 26.6246185
28: -5.4694290, 23.8641396, -5.4874020, 23.8994350, -29.1025124, 29.1478043
29: -6.2508945, 24.1936779, -6.2901478, 24.2026863, -26.0571518, 26.1039696
30: -12.4776363, 22.0811386, -12.5159750, 22.0928078, -30.3552628, 30.4114380
31: -17.1999245, 27.2023945, -17.2259407, 27.2333908, -39.2851410, 39.2765198
32: -70.1103210, -28.9625626, -70.1558609, -28.9553680, -30.7002907, 30.7287502
33: -110.2301025, -44.8361893, -110.2951202, -44.8027344, -45.2364883, 45.2426529
34: -88.6915741, -42.1829758, -88.7293320, -42.1692123, -28.9958191, 29.0038147
35: -66.4970703, -20.9542198, -66.5423050, -20.9393806, -30.7522430, 30.7815742
36: -60.7597351, -16.1380081, -60.7977409, -16.1085377, -29.4491348, 29.4665184
37: -121.2923431, -55.2783089, -121.3598328, -55.2384834, -46.0081940, 46.0306931
38: -70.6162109, -15.7036457, -70.6560516, -15.6664371, -39.6815643, 39.6565857
39: -110.9680023, -44.5009270, -111.0270844, -44.4626579, -41.3346405, 41.3891373
40: -114.8009644, -64.8559113, -114.8185043, -64.8419037, -35.6791763, 35.5577850
41: -87.1319275, -42.6120377, -87.1597366, -42.5801735, -29.3756409, 29.4082298
42: -72.5140381, -36.0040627, -72.5390930, -35.9880600, -30.0495224, 30.0635300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1725108, upper bound: 24.3232072
time: 43.03 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2461011, upper bound: 24.3257100
time: 36.85 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -22.2148094, 15.9236584, -22.1650715, 15.9137125, -33.8029861, 33.7253685
1: -2.0487447, 26.0800247, -1.9536166, 26.0776119, -23.8592758, 23.7637482
2: 1.4031100, 24.3141632, 1.4664054, 24.3098679, -20.5391579, 20.4856377
3: -12.9313755, 14.6409206, -12.9202204, 14.6435366, -23.6331673, 23.6257057
4: -1.9416330, 26.2360687, -1.8577106, 26.2327213, -24.4589806, 24.3579903
5: 0.2362552, 23.3430405, 0.2669706, 23.3377800, -20.7854195, 20.7127762
6: -74.5499573, -33.9961243, -74.5328140, -34.0674896, -29.2501068, 29.2329178
7: -5.0645895, 19.2751198, -5.0120664, 19.2717361, -20.1411858, 20.0697498
8: -9.0096197, 20.5503998, -8.8732681, 20.5656281, -28.1375351, 27.9740181
9: -26.2847900, 10.9458818, -26.2487106, 10.9312277, -30.6651306, 30.6149597
10: -35.3776703, 10.9609623, -35.3833923, 10.9359798, -37.1731873, 37.1118469
11: -25.0178909, 9.9715633, -25.0088234, 9.9473209, -27.5498962, 27.5928345
12: -42.5500069, -5.8953509, -42.5763817, -5.9846954, -26.1044388, 26.1040688
13: -38.5304642, 20.3386078, -38.5492172, 20.1833153, -43.2827377, 43.5114098
14: 1.8095179, 49.0682106, 1.8605061, 49.0532799, -39.5597000, 39.4807281
15: -17.0743694, 19.4994926, -16.9785042, 19.4822884, -32.0222931, 31.9333572
16: -38.1756287, 6.7231560, -38.0715027, 6.7118235, -34.3876572, 34.2755852
17: 9.9083853, 55.1475945, 9.8950987, 55.0475121, -36.0056839, 36.0237770
18: -8.2689304, 38.7742271, -8.1802425, 38.7690468, -41.5732269, 41.3483658
19: -18.1395874, 17.4245434, -18.1335106, 17.3917217, -30.6079330, 30.7258186
20: -16.1578598, 18.7611179, -16.1467705, 18.7195053, -31.8790283, 31.9285660
21: -16.0819988, 19.3303585, -16.0603237, 19.2927361, -33.3961945, 33.5016327
22: -2.4261456, 31.5513802, -2.4108272, 31.4734612, -27.6131477, 27.6828995
23: -14.9135628, 15.3030090, -14.9025698, 15.2969580, -28.5479050, 28.7095032
24: -8.8208084, 29.2559528, -8.8074074, 29.2566700, -33.4868698, 33.5131874
25: 1.1798172, 39.0447311, 1.2031822, 39.0006790, -34.1707535, 34.3439407
26: -16.2868500, 24.3096771, -16.2479973, 24.3267536, -36.6492767, 36.6827393
27: -14.4474115, 21.6852303, -14.3471861, 21.7156143, -26.7017670, 26.6733322
28: -5.5065145, 23.9201374, -5.4937630, 23.9017982, -29.1359711, 29.2610016
29: -6.3208761, 24.2684155, -6.3085814, 24.2041302, -26.1275368, 26.2002106
30: -12.5620871, 22.1899338, -12.5348616, 22.0950241, -30.4360580, 30.5415344
31: -17.2518234, 27.2413177, -17.2321167, 27.2345390, -39.3216629, 39.3897171
32: -70.1809845, -28.8570271, -70.1764526, -28.9533920, -30.7710915, 30.8594799
33: -110.3394547, -44.6490479, -110.3312531, -44.8007202, -45.3243256, 45.4907684
34: -88.7619247, -42.0491524, -88.7502899, -42.1684570, -29.0573349, 29.1606064
35: -66.5698853, -20.8338509, -66.5657043, -20.9388504, -30.8142929, 30.9389229
36: -60.8244934, -16.0323772, -60.8187256, -16.1077709, -29.5079041, 29.5985298
37: -121.4045868, -55.1894150, -121.3937225, -55.2377014, -46.1104050, 46.1777954
38: -70.6883392, -15.6386080, -70.6747437, -15.6658335, -39.7507324, 39.7559128
39: -111.0635910, -44.3579712, -111.0579681, -44.4618301, -41.4162827, 41.5797882
40: -114.8565674, -64.8264923, -114.8261108, -64.8417969, -35.7713623, 35.5802078
41: -87.1928253, -42.5400314, -87.1715622, -42.5790100, -29.4458847, 29.4862442
42: -72.5783310, -35.9438248, -72.5503235, -35.9869614, -30.1102905, 30.1365509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 169

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2152555, upper bound: 24.3232735
time: 59.23 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2888480, upper bound: 24.3257339
time: 42.59 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.2048531, 15.9771252, -22.2803345, 15.9264927, -33.7631912, 33.8756866
1: -1.9708443, 26.1122055, -2.0753179, 26.0827217, -23.7810440, 23.9172096
2: 1.4598162, 24.3232727, 1.3990915, 24.3207912, -20.4999161, 20.5758038
3: -12.9342136, 14.6913462, -12.9642458, 14.6524973, -23.6133308, 23.7168961
4: -1.8654761, 26.2450943, -1.9455097, 26.2494087, -24.3749352, 24.4846992
5: 0.2511773, 23.3759460, 0.2087607, 23.3479366, -20.7374573, 20.8527298
6: -74.5676193, -34.0551758, -74.5583725, -33.9784546, -29.2812042, 29.2684898
7: -5.0149832, 19.2821922, -5.0673108, 19.2805672, -20.0740814, 20.1534252
8: -8.9176121, 20.5885258, -9.0129910, 20.5816841, -28.0524025, 28.1912918
9: -26.2704315, 11.0022125, -26.3347893, 10.9576321, -30.6558647, 30.7877235
10: -35.4079819, 11.0193577, -35.4292793, 10.9729271, -37.2019424, 37.3086243
11: -25.0482655, 9.9633045, -25.0271816, 10.0046501, -27.6784134, 27.5722046
12: -42.6020317, -5.9138741, -42.6080475, -5.8929582, -26.2033501, 26.2341881
13: -38.5719719, 20.2664146, -38.5815964, 20.3497734, -43.5200500, 43.4206200
14: 1.8118095, 49.1495628, 1.7260284, 49.0719223, -39.5533600, 39.7436447
15: -16.9998131, 19.5140800, -17.0988579, 19.5084305, -31.9477234, 32.0826111
16: -38.0946503, 6.7653604, -38.2175789, 6.7280917, -34.2996292, 34.4865685
17: 9.8486643, 55.1305733, 9.8143072, 55.1507950, -36.1492767, 36.1897202
18: -8.2021027, 38.7712288, -8.2856531, 38.7760811, -41.3592453, 41.5151520
19: -18.2287922, 17.4149590, -18.1457329, 17.4839268, -30.8821869, 30.5987167
20: -16.2260437, 18.7516155, -16.1605225, 18.8215923, -32.0693130, 31.9137802
21: -16.1343918, 19.3184757, -16.0890179, 19.3836632, -33.6301575, 33.4234085
22: -2.4564075, 31.4836845, -2.4305563, 31.5737114, -27.7515182, 27.6168060
23: -14.9956131, 15.3274269, -14.9201508, 15.3617058, -28.8623886, 28.6133652
24: -8.8827562, 29.2785263, -8.8330965, 29.3119736, -33.6466141, 33.4589500
25: 1.1463509, 39.0211029, 1.1710548, 39.0878220, -34.4484634, 34.1834488
26: -16.3336697, 24.3539639, -16.2934952, 24.3663578, -36.8246155, 36.6764374
27: -14.4604053, 21.7416573, -14.4520006, 21.7497139, -26.8538780, 26.8546562
28: -5.5741568, 23.9305096, -5.5112362, 23.9724884, -29.3948631, 29.1938438
29: -6.3398504, 24.2114449, -6.3290262, 24.2852306, -26.2469406, 26.1383953
30: -12.5728197, 22.1193352, -12.5692596, 22.2243881, -30.5806732, 30.4650345
31: -17.3052082, 27.2529411, -17.2658520, 27.2895050, -39.5123901, 39.2563171
32: -70.1912613, -28.9434128, -70.1893311, -28.8489037, -30.8581009, 30.7931442
33: -110.4042969, -44.7778244, -110.3461456, -44.6002312, -45.6139755, 45.3314056
34: -88.7608795, -42.1625023, -88.7658157, -42.0366898, -29.1782570, 29.0771980
35: -66.5875397, -20.9287968, -66.5782318, -20.8135223, -30.9853592, 30.8298645
36: -60.8845863, -16.0900822, -60.8310318, -15.9900599, -29.7086639, 29.5355949
37: -121.4966736, -55.2105179, -121.4231873, -55.1247711, -46.3450089, 46.0810432
38: -70.7568207, -15.6436081, -70.6978836, -15.5832930, -39.8805847, 39.7332077
39: -111.1513214, -44.4359169, -111.0701752, -44.2954941, -41.7356873, 41.4322128
40: -114.8701401, -64.8268585, -114.8645935, -64.8030701, -35.6101837, 35.7242966
41: -87.2480392, -42.5570412, -87.2002258, -42.4892273, -29.6221313, 29.5305157
42: -72.5915298, -35.9688568, -72.5846863, -35.9143219, -30.2165680, 30.1371918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 169

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3232734, upper bound: 24.2521762
time: 50.69 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3257338, upper bound: 24.3257337
time: 46.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 99.23 seconds
IS_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2070735, upper bound: 24.2521524
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2101070, upper bound: 24.3257100
IS_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.1500930, upper bound: 24.2521764
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2528311, upper bound: 24.3257339
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.1725108, upper bound: 24.3232072
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2461011, upper bound: 24.3257100
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2152555, upper bound: 24.3232735
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.2888480, upper bound: 24.3257339
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.3232734, upper bound: 24.2521762
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 99.23
Output dim: 17, lower bound: -24.3257338, upper bound: 24.3257337

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.1001396, 15.8631372, -22.1148014, 15.8756676, -33.6721077, 33.5875854
1: -1.9247088, 26.0246468, -1.9215574, 26.0441952, -23.7083626, 23.6839027
2: 1.4826593, 24.2598343, 1.5006030, 24.2759933, -20.4249649, 20.4069862
3: -12.8922558, 14.5979328, -12.9110508, 14.6131153, -23.5318642, 23.5892963
4: -1.8274126, 26.1715431, -1.7816885, 26.1915855, -24.3001175, 24.2327042
5: 0.2905059, 23.3066540, 0.2771873, 23.3182640, -20.6630173, 20.7012577
6: -74.4623871, -34.0965195, -74.4869537, -34.1252480, -29.0617523, 29.1430435
7: -5.0008197, 19.2336121, -4.9981799, 19.2456818, -20.0196609, 20.0266075
8: -8.8546028, 20.4920788, -8.8320131, 20.5308990, -27.9354134, 27.9067001
9: -26.2138023, 10.8615179, -26.2384453, 10.8849745, -30.5170670, 30.5212059
10: -35.3430939, 10.8799477, -35.3601799, 10.8745155, -37.0379944, 37.0030365
11: -24.9821587, 9.9043236, -24.9830627, 9.8791733, -27.4500198, 27.4515572
12: -42.4566498, -6.0104566, -42.5123444, -6.0579967, -25.8946228, 25.9765739
13: -38.4459152, 20.1394520, -38.4982529, 20.0994720, -43.1279297, 43.2428856
14: 1.9426222, 49.0049744, 1.9154072, 48.9994659, -39.3497124, 39.3895302
15: -16.9366970, 19.4221840, -16.9018974, 19.4350567, -31.8350143, 31.7507095
16: -38.0435562, 6.6449080, -38.0560532, 6.6744604, -34.1675034, 34.1796570
17: 10.0671263, 55.0272255, 9.9838905, 54.9935188, -35.7754745, 35.8213501
18: -8.1317892, 38.7020645, -8.0883493, 38.7195206, -41.3918915, 41.2017517
19: -18.1059608, 17.3514004, -18.1066437, 17.3848648, -30.6338043, 30.5061646
20: -16.1064682, 18.6649742, -16.1135483, 18.6780510, -31.7766418, 31.7862129
21: -16.0303955, 19.2465057, -16.0308990, 19.2609978, -33.3365936, 33.3205948
22: -2.3566318, 31.4579391, -2.3724794, 31.4731331, -27.5469551, 27.5333786
23: -14.8786526, 15.2544098, -14.8848152, 15.2860775, -28.5655823, 28.5197029
24: -8.7751284, 29.2164211, -8.7669353, 29.2445335, -33.4724350, 33.3455887
25: 1.2362642, 38.9639015, 1.2281523, 38.9831009, -34.1710205, 34.1026459
26: -16.2201233, 24.2820702, -16.2151623, 24.3239670, -36.6028976, 36.5417557
27: -14.3168488, 21.6338367, -14.2867651, 21.6855068, -26.5285263, 26.5769291
28: -5.4675150, 23.8621559, -5.4746199, 23.8936768, -29.1099319, 29.0943604
29: -6.2497950, 24.1808319, -6.2680073, 24.1756744, -26.0462570, 26.0642738
30: -12.4742012, 22.0462036, -12.4884567, 22.0211830, -30.2592316, 30.3622169
31: -17.1974888, 27.2004070, -17.1973858, 27.2281837, -39.3535461, 39.1681366
32: -70.1065674, -28.9765301, -70.1360016, -28.9999695, -30.6148911, 30.7281761
33: -110.2088165, -44.8408623, -110.2467957, -44.8339539, -45.2128296, 45.1850891
34: -88.6760254, -42.1831779, -88.6893234, -42.1961632, -28.9328613, 28.9828835
35: -66.4840393, -20.9572010, -66.5096512, -20.9625206, -30.7363434, 30.7361221
36: -60.7564392, -16.1379051, -60.7830544, -16.1174583, -29.4295959, 29.4602890
37: -121.2697144, -55.2845917, -121.2978821, -55.2713127, -45.9884644, 45.9209442
38: -70.6056519, -15.7030773, -70.6266327, -15.6794472, -39.6651993, 39.6313248
39: -110.9525299, -44.5057983, -110.9875183, -44.4925690, -41.3628845, 41.3074646
40: -114.7794800, -64.8608627, -114.7705994, -64.8710098, -35.5951920, 35.5297241
41: -87.1183319, -42.6174355, -87.1281662, -42.6075592, -29.3178635, 29.4034767
42: -72.5127716, -36.0248260, -72.5242996, -36.0435562, -29.9967194, 30.0197029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2082007, upper bound: 24.2570276
time: 39.80 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2087457, upper bound: 24.3243366
time: 44.59 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.1953888, 15.9217186, -22.1185455, 15.8926811, -33.7859879, 33.6499290
1: -2.0340641, 26.0810280, -1.9226718, 26.0625000, -23.8370132, 23.7373962
2: 1.4190338, 24.3128281, 1.4997020, 24.2919846, -20.5063362, 20.4607773
3: -12.9266376, 14.6443453, -12.9125223, 14.6254597, -23.5820770, 23.6382942
4: -1.9091899, 26.2343788, -1.7832108, 26.2103977, -24.4053192, 24.2950287
5: 0.2401509, 23.3436241, 0.2760749, 23.3279934, -20.7284126, 20.7388039
6: -74.5460205, -34.0208054, -74.5076752, -34.1250305, -29.1387177, 29.2257080
7: -5.0604506, 19.2743492, -4.9996939, 19.2570076, -20.0925751, 20.0692787
8: -8.9928083, 20.5485725, -8.8332500, 20.5501595, -28.1030731, 27.9621315
9: -26.2865601, 10.9346352, -26.2392540, 10.9046106, -30.6209183, 30.5944633
10: -35.3818283, 10.9389906, -35.3622589, 10.8845263, -37.1239853, 37.0660896
11: -25.0150719, 9.9390335, -24.9861755, 9.8800392, -27.4890366, 27.5254555
12: -42.5460548, -5.9257817, -42.5421295, -6.0570841, -25.9730110, 26.0970497
13: -38.5289993, 20.3039818, -38.5247421, 20.1035061, -43.2027740, 43.4428902
14: 1.8166256, 49.0499344, 1.9067917, 49.0118484, -39.4842567, 39.4408798
15: -17.0372562, 19.4935074, -16.9030418, 19.4526596, -31.9739990, 31.8177299
16: -38.1809921, 6.7128863, -38.0574837, 6.6932735, -34.3385162, 34.2431107
17: 9.9161539, 55.1225471, 9.9437275, 54.9938354, -35.9186249, 35.9628220
18: -8.2331543, 38.7724915, -8.0926027, 38.7432632, -41.5215073, 41.2725754
19: -18.1453075, 17.4205647, -18.1130867, 17.3856392, -30.6690903, 30.6175804
20: -16.1557083, 18.7426529, -16.1265793, 18.6803169, -31.8256760, 31.8947945
21: -16.0831871, 19.3158588, -16.0382118, 19.2630196, -33.3764877, 33.4522514
22: -2.4181805, 31.5521088, -2.3891754, 31.4748440, -27.6091156, 27.6506882
23: -14.9119291, 15.2994986, -14.8900776, 15.2875061, -28.5892792, 28.6304398
24: -8.8083258, 29.2545872, -8.7695160, 29.2462807, -33.5027466, 33.4342651
25: 1.1840181, 39.0374069, 1.2225170, 38.9858589, -34.2093430, 34.2600861
26: -16.2853737, 24.3085594, -16.2222023, 24.3261623, -36.6516190, 36.6249275
27: -14.4259729, 21.6841240, -14.2903881, 21.7011700, -26.6623383, 26.6256256
28: -5.5046082, 23.9181786, -5.4809799, 23.8960438, -29.1434021, 29.2076035
29: -6.3197441, 24.2555485, -6.2864552, 24.1771393, -26.1166763, 26.1605320
30: -12.5587177, 22.1549568, -12.5073404, 22.0233860, -30.3400421, 30.4923096
31: -17.2493000, 27.2393322, -17.2035904, 27.2293205, -39.3900070, 39.2813110
32: -70.1772614, -28.8710480, -70.1565704, -28.9980812, -30.6856880, 30.8589039
33: -110.3181763, -44.6537170, -110.2830124, -44.8319130, -45.3004608, 45.4332581
34: -88.7463150, -42.0493431, -88.7102966, -42.1953850, -28.9943542, 29.1397095
35: -66.5568619, -20.8367996, -66.5330429, -20.9619732, -30.7983475, 30.8935509
36: -60.8211708, -16.0323086, -60.8040466, -16.1166992, -29.4883728, 29.5923309
37: -121.3819733, -55.1956863, -121.3317795, -55.2705498, -46.0906067, 46.0680618
38: -70.6777954, -15.6380272, -70.6453323, -15.6788578, -39.7343445, 39.7306519
39: -111.0481339, -44.3628235, -111.0183563, -44.4918442, -41.4444733, 41.4981155
40: -114.8350601, -64.8314743, -114.7781830, -64.8708344, -35.6873703, 35.5521431
41: -87.1792068, -42.5454636, -87.1399765, -42.6063881, -29.3881531, 29.4815254
42: -72.5770645, -35.9645348, -72.5355606, -36.0424194, -30.0574951, 30.0927467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2509244, upper bound: 24.2570501
time: 50.62 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2514694, upper bound: 24.3243647
time: 41.20 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -22.0892925, 15.8272276, -22.1594162, 15.8784962, -33.6435165, 33.6249390
1: -1.9317503, 26.0161972, -1.9514923, 26.0564117, -23.7129974, 23.7023201
2: 1.4783151, 24.2512550, 1.4691343, 24.2896652, -20.4426842, 20.4208450
3: -12.8773594, 14.5798893, -12.9104843, 14.6263971, -23.5571709, 23.5535870
4: -1.8223047, 26.1573906, -1.8527777, 26.2068977, -24.3078117, 24.2773399
5: 0.3037696, 23.2977657, 0.2745800, 23.3250313, -20.6946068, 20.6628761
6: -74.4444275, -34.1074867, -74.5018463, -34.0694199, -29.1475487, 29.1014862
7: -4.9822559, 19.2126808, -4.9999580, 19.2570248, -20.0378876, 19.9921379
8: -8.8538418, 20.4851608, -8.8693619, 20.5427094, -27.9498444, 27.8970680
9: -26.1977310, 10.8300638, -26.2463760, 10.8942938, -30.5190659, 30.4915810
10: -35.3258591, 10.8519831, -35.3790932, 10.9086742, -37.0428925, 36.9931145
11: -24.9656944, 9.8943901, -24.9970798, 9.9441319, -27.4855309, 27.4586830
12: -42.4259071, -6.0155172, -42.5304375, -5.9873896, -25.9861221, 25.9281273
13: -38.4145050, 20.1286430, -38.5069199, 20.1762466, -43.1712494, 43.2563782
14: 1.9846401, 49.0048599, 1.8880243, 49.0392685, -39.3727112, 39.3934555
15: -16.9076347, 19.3883953, -16.9750824, 19.4462013, -31.7862778, 31.8181458
16: -38.0016441, 6.6061592, -38.0688171, 6.6752310, -34.1674042, 34.1620789
17: 10.1407623, 55.0114517, 9.9715357, 55.0467377, -35.7781296, 35.7964058
18: -8.1171598, 38.6734505, -8.1709976, 38.7304840, -41.3770981, 41.2423706
19: -18.0798187, 17.3449383, -18.1232643, 17.3876324, -30.5572357, 30.5941620
20: -16.0804367, 18.6575966, -16.1203213, 18.7158318, -31.7983704, 31.7773132
21: -16.0144196, 19.2481308, -16.0480843, 19.2893524, -33.3396759, 33.3320084
22: -2.3405561, 31.4554634, -2.3895931, 31.4714584, -27.5254135, 27.5564690
23: -14.8684721, 15.2424793, -14.8947210, 15.2894211, -28.5184479, 28.5633545
24: -8.7691832, 29.2135639, -8.8018303, 29.2532959, -33.4323425, 33.4148483
25: 1.2551756, 38.9637184, 1.2164459, 38.9961739, -34.1079407, 34.1549377
26: -16.1890144, 24.2739944, -16.2332802, 24.3202610, -36.5710449, 36.5779190
27: -14.3154955, 21.6270199, -14.3416195, 21.6967106, -26.5415497, 26.6152287
28: -5.4578485, 23.8574066, -5.4835148, 23.8970356, -29.0872116, 29.1140099
29: -6.2382851, 24.1893425, -6.2855625, 24.2013893, -26.0437775, 26.0934410
30: -12.4367418, 22.0284576, -12.4973631, 22.0897789, -30.3106766, 30.3379898
31: -17.1780453, 27.1939888, -17.2205372, 27.2297630, -39.2602005, 39.2612305
32: -70.0934906, -28.9948959, -70.1481934, -28.9577579, -30.6785812, 30.6838207
33: -110.2126389, -44.8543243, -110.2914505, -44.8104324, -45.2126465, 45.2150726
34: -88.6800766, -42.1999359, -88.7279053, -42.1746483, -28.9740448, 28.9840088
35: -66.4809723, -20.9698067, -66.5393219, -20.9458752, -30.7313461, 30.7603149
36: -60.7495995, -16.1443996, -60.7940292, -16.1094017, -29.4362946, 29.4554825
37: -121.2372284, -55.3387833, -121.3535309, -55.2665482, -45.9282227, 45.9591293
38: -70.5884171, -15.7091055, -70.6458969, -15.6681070, -39.6505737, 39.6426163
39: -110.9515839, -44.5207825, -111.0239258, -44.4703445, -41.3147430, 41.3610306
40: -114.7629166, -64.9040756, -114.8153381, -64.8625336, -35.6285172, 35.5093689
41: -87.1115875, -42.6487808, -87.1589508, -42.5937881, -29.3524017, 29.3721981
42: -72.5097656, -36.0333557, -72.5377579, -35.9915466, -30.0430832, 30.0137825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1705793, upper bound: 24.2545369
time: 43.27 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.1711362, upper bound: 24.3218225
time: 46.10 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -22.1191063, 15.8639698, -22.1611252, 15.8961678, -33.6866989, 33.6446571
1: -1.9391961, 26.0231018, -1.9523659, 26.0590096, -23.7349625, 23.7073517
2: 1.4674711, 24.2596836, 1.4676816, 24.2931290, -20.4557724, 20.4287033
3: -12.8939390, 14.5939503, -12.9172153, 14.6309004, -23.5737228, 23.5741901
4: -1.8584378, 26.1725693, -1.8554718, 26.2135658, -24.3523331, 24.2939682
5: 0.2900896, 23.3056202, 0.2698598, 23.3278313, -20.7276993, 20.6696968
6: -74.4606781, -34.0723343, -74.5091095, -34.0679321, -29.1453705, 29.1478424
7: -4.9996567, 19.2337723, -5.0079350, 19.2600708, -20.0452156, 20.0248680
8: -8.8703537, 20.4922791, -8.8715086, 20.5454941, -27.9617348, 27.9266014
9: -26.2115154, 10.8657579, -26.2476482, 10.9081202, -30.5597916, 30.5147934
10: -35.3382416, 10.8936663, -35.3809853, 10.9218798, -37.0849762, 37.0197678
11: -24.9815025, 9.9359989, -25.0038242, 9.9460087, -27.4907837, 27.5176888
12: -42.4577560, -5.9806919, -42.5449905, -5.9858804, -26.0012321, 25.9825287
13: -38.4453888, 20.1732521, -38.5217514, 20.1788311, -43.1750412, 43.3085632
14: 1.9369221, 49.0227661, 1.8698244, 49.0406799, -39.4059525, 39.4280624
15: -16.9731064, 19.4204674, -16.9769764, 19.4605331, -31.8813934, 31.8251877
16: -38.0377731, 6.6538591, -38.0698662, 6.6923628, -34.2144928, 34.1788101
17: 10.0643682, 55.0520973, 9.9377184, 55.0471268, -35.8107300, 35.8811226
18: -8.1666641, 38.7019348, -8.1754961, 38.7444267, -41.4411316, 41.2697601
19: -18.0994987, 17.3526859, -18.1266899, 17.3896446, -30.5635223, 30.6164093
20: -16.1069298, 18.6831112, -16.1326332, 18.7171059, -31.8157272, 31.8192329
21: -16.0276222, 19.2607365, -16.0522499, 19.2905712, -33.3507843, 33.3791733
22: -2.3632469, 31.4571762, -2.3934183, 31.4717293, -27.5490265, 27.5639877
23: -14.8799543, 15.2551279, -14.8971539, 15.2941628, -28.5121994, 28.6205101
24: -8.7869987, 29.2168465, -8.8045492, 29.2543831, -33.4549408, 33.4192085
25: 1.2346058, 38.9706993, 1.2100983, 38.9976654, -34.1279526, 34.1902542
26: -16.2203751, 24.2814560, -16.2402821, 24.3236942, -36.5929337, 36.5980644
27: -14.3373270, 21.6334915, -14.3430614, 21.6992226, -26.5659065, 26.6191444
28: -5.4681396, 23.8630733, -5.4867468, 23.8988953, -29.0953560, 29.1576614
29: -6.2492046, 24.1930981, -6.2892876, 24.2023888, -26.0547333, 26.1021767
30: -12.4738884, 22.0802422, -12.5141153, 22.0923519, -30.3228645, 30.4093094
31: -17.1987934, 27.2004719, -17.2253532, 27.2324390, -39.2816467, 39.2693481
32: -70.1063995, -28.9629955, -70.1538239, -28.9555779, -30.6814651, 30.7269611
33: -110.2295151, -44.8402863, -110.2948074, -44.8047943, -45.2255173, 45.2411575
34: -88.6909485, -42.1851501, -88.7290115, -42.1703186, -28.9933853, 28.9920845
35: -66.4965515, -20.9575768, -66.5420532, -20.9409981, -30.7470245, 30.7752495
36: -60.7581978, -16.1382027, -60.7969513, -16.1086197, -29.4477768, 29.4651680
37: -121.2917786, -55.2845078, -121.3595428, -55.2415886, -46.0034332, 45.9715805
38: -70.6124115, -15.7042494, -70.6541672, -15.6667366, -39.6865234, 39.6516571
39: -110.9668198, -44.5055466, -111.0265656, -44.4650650, -41.3270569, 41.3850098
40: -114.8002396, -64.8592758, -114.8181610, -64.8436050, -35.6758423, 35.5194626
41: -87.1313477, -42.6146698, -87.1594162, -42.5815163, -29.3723564, 29.3918438
42: -72.5137482, -36.0050621, -72.5389252, -35.9885674, -30.0426254, 30.0744476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2441925, upper bound: 24.2570276
time: 26.04 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2447324, upper bound: 24.3243366
time: 28.02 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -22.1845226, 15.8858299, -22.1631355, 15.8955240, -33.7574234, 33.6872826
1: -2.0411108, 26.0725880, -1.9525881, 26.0747395, -23.8416557, 23.7557869
2: 1.4146924, 24.3042374, 1.4682488, 24.3056564, -20.5240746, 20.4746399
3: -12.9117508, 14.6262789, -12.9119282, 14.6387539, -23.6074028, 23.6025734
4: -1.9040782, 26.2202187, -1.8543241, 26.2257309, -24.4129753, 24.3396454
5: 0.2534242, 23.3347359, 0.2734504, 23.3347626, -20.7599983, 20.7004185
6: -74.5280075, -34.0317154, -74.5225601, -34.0692673, -29.2245674, 29.1841621
7: -5.0419087, 19.2533760, -5.0014648, 19.2683907, -20.1107979, 20.0348091
8: -8.9920492, 20.5416451, -8.8705730, 20.5619736, -28.1174927, 27.9525108
9: -26.2705364, 10.9031744, -26.2472000, 10.9139338, -30.6229248, 30.5648537
10: -35.3646240, 10.9110870, -35.3811646, 10.9185781, -37.1288834, 37.0562134
11: -24.9985352, 9.9290771, -25.0001945, 9.9449854, -27.5245590, 27.5326004
12: -42.5153198, -5.9308286, -42.5602608, -5.9865084, -26.0644798, 26.0486031
13: -38.4975967, 20.2931175, -38.5334816, 20.1803627, -43.2460861, 43.4564056
14: 1.8585548, 49.0497627, 1.8794069, 49.0516129, -39.5073166, 39.4448547
15: -17.0081711, 19.4596958, -16.9762745, 19.4637756, -31.9252777, 31.8851967
16: -38.1391373, 6.6741791, -38.0703049, 6.6940236, -34.3384132, 34.2255669
17: 9.9897747, 55.1067772, 9.9314194, 55.0470810, -35.9212799, 35.9378662
18: -8.2184982, 38.7439575, -8.1752758, 38.7542267, -41.5067368, 41.3131866
19: -18.1190968, 17.4141235, -18.1297054, 17.3883953, -30.5925293, 30.7056046
20: -16.1296883, 18.7352943, -16.1333599, 18.7180786, -31.8474045, 31.8859291
21: -16.0672302, 19.3174877, -16.0553951, 19.2914009, -33.3795395, 33.4636230
22: -2.4021301, 31.5496178, -2.4063225, 31.4731808, -27.5876770, 27.6737785
23: -14.9017410, 15.2876015, -14.8999815, 15.2908363, -28.5420837, 28.6741028
24: -8.8023605, 29.2517738, -8.8043909, 29.2550621, -33.4626007, 33.5034943
25: 1.2029552, 39.0372391, 1.2107811, 38.9988976, -34.1463394, 34.3123627
26: -16.2542305, 24.3005066, -16.2403450, 24.3224754, -36.6197815, 36.6611366
27: -14.4246140, 21.6773071, -14.3452158, 21.7123547, -26.6753464, 26.6639366
28: -5.4949379, 23.9134407, -5.4898586, 23.8994102, -29.1206818, 29.2272339
29: -6.3082476, 24.2640800, -6.3040018, 24.2028561, -26.1141586, 26.1897182
30: -12.5212774, 22.1372509, -12.5162563, 22.0919876, -30.3914948, 30.4681244
31: -17.2298660, 27.2329407, -17.2267342, 27.2309113, -39.2966461, 39.3743973
32: -70.1641998, -28.8894253, -70.1687927, -28.9558411, -30.7494278, 30.8145638
33: -110.3218765, -44.6671448, -110.3276062, -44.8083954, -45.3003998, 45.4631805
34: -88.7504120, -42.0660820, -88.7488556, -42.1738930, -29.0355682, 29.1408386
35: -66.5538025, -20.8493786, -66.5627136, -20.9453773, -30.7933731, 30.9177132
36: -60.8143349, -16.0387764, -60.8150406, -16.1086025, -29.4950714, 29.5875244
37: -121.3494568, -55.2498627, -121.3874664, -55.2657547, -46.0303726, 46.1062851
38: -70.6604614, -15.6439524, -70.6646042, -15.6675663, -39.7197113, 39.7418671
39: -111.0472107, -44.3778877, -111.0547180, -44.4695587, -41.3964005, 41.5516968
40: -114.8184814, -64.8746719, -114.8229218, -64.8623962, -35.7207718, 35.5317650
41: -87.1724777, -42.5767670, -87.1707611, -42.5926323, -29.4226646, 29.4502316
42: -72.5740662, -35.9730568, -72.5490265, -35.9904594, -30.1038208, 30.0868149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2133305, upper bound: 24.2545968
time: 113.29 seconds

## Relational analysis of IS_A1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2138847, upper bound: 24.3218920
time: 40.68 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -22.2143784, 15.9225731, -22.1648712, 15.9131870, -33.8005905, 33.7069397
1: -2.0485713, 26.0794888, -1.9534931, 26.0773430, -23.8636208, 23.7608414
2: 1.4038408, 24.3126831, 1.4667847, 24.3091145, -20.5371666, 20.4825211
3: -12.9283257, 14.6403837, -12.9186687, 14.6432705, -23.6239624, 23.6231995
4: -1.9402468, 26.2354126, -1.8570068, 26.2323952, -24.4575119, 24.3562622
5: 0.2397504, 23.3425827, 0.2687497, 23.3375607, -20.7930946, 20.7072449
6: -74.5443497, -33.9966049, -74.5297775, -34.0677414, -29.2223930, 29.2305031
7: -5.0592957, 19.2744827, -5.0094585, 19.2714252, -20.1181641, 20.0675659
8: -9.0085535, 20.5487690, -8.8727283, 20.5647545, -28.1293831, 27.9820480
9: -26.2843170, 10.9388838, -26.2484398, 10.9277382, -30.6636543, 30.5880547
10: -35.3770218, 10.9526711, -35.3830719, 10.9318609, -37.1709671, 37.0828400
11: -25.0143681, 9.9707298, -25.0069160, 9.9468508, -27.5297661, 27.5916367
12: -42.5471878, -5.8960061, -42.5747833, -5.9850407, -26.0796051, 26.1030235
13: -38.5284996, 20.3377590, -38.5482254, 20.1829319, -43.2498703, 43.5086060
14: 1.8108940, 49.0676956, 1.8611660, 49.0530319, -39.5405045, 39.4794617
15: -17.0736389, 19.4917660, -16.9781570, 19.4781227, -32.0204010, 31.8922119
16: -38.1752357, 6.7219219, -38.0713387, 6.7112041, -34.3854904, 34.2423401
17: 9.9133892, 55.1474533, 9.8975773, 55.0474930, -35.9538269, 36.0225906
18: -8.2680435, 38.7724228, -8.1797876, 38.7681351, -41.5707703, 41.3405762
19: -18.1389046, 17.4218712, -18.1331310, 17.3903904, -30.5988235, 30.7278290
20: -16.1561623, 18.7608013, -16.1456566, 18.7193680, -31.8647766, 31.9278717
21: -16.0804672, 19.3300838, -16.0595627, 19.2926102, -33.3906555, 33.5108109
22: -2.4247923, 31.5513287, -2.4101629, 31.4734516, -27.6112900, 27.6813164
23: -14.9132290, 15.3002319, -14.9024143, 15.2955732, -28.5358734, 28.7312851
24: -8.8202000, 29.2550316, -8.8071041, 29.2561798, -33.4852524, 33.5078964
25: 1.1823578, 39.0442200, 1.2044678, 39.0004120, -34.1663437, 34.3477249
26: -16.2856388, 24.3079624, -16.2473412, 24.3259087, -36.6416931, 36.6812706
27: -14.4464512, 21.6837749, -14.3466883, 21.7148952, -26.6996994, 26.6678543
28: -5.5052471, 23.9190750, -5.4930925, 23.9012680, -29.1288147, 29.2708855
29: -6.3191719, 24.2678452, -6.3077288, 24.2038307, -26.1251259, 26.1984444
30: -12.5583849, 22.1890068, -12.5330124, 22.0945740, -30.4036713, 30.5394211
31: -17.2506428, 27.2393951, -17.2315350, 27.2335720, -39.3181610, 39.3824997
32: -70.1771011, -28.8575115, -70.1744232, -28.9536743, -30.7522888, 30.8576775
33: -110.3388519, -44.6530991, -110.3309708, -44.8027763, -45.3133469, 45.4893494
34: -88.7612305, -42.0512772, -88.7499390, -42.1695633, -29.0549126, 29.1489067
35: -66.5693436, -20.8371735, -66.5654373, -20.9405289, -30.8090515, 30.9326668
36: -60.8229485, -16.0326061, -60.8179169, -16.1078529, -29.5065308, 29.5971870
37: -121.4040833, -55.1956139, -121.3934631, -55.2408562, -46.1056442, 46.1187897
38: -70.6845016, -15.6391706, -70.6728745, -15.6661625, -39.7556992, 39.7509918
39: -111.0624390, -44.3626862, -111.0574188, -44.4643631, -41.4087372, 41.5756912
40: -114.8557510, -64.8299103, -114.8257446, -64.8434906, -35.7680359, 35.5418587
41: -87.1922684, -42.5426331, -87.1712265, -42.5803604, -29.4426041, 29.4699059
42: -72.5780258, -35.9448280, -72.5501862, -35.9874496, -30.1033630, 30.1475029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2869402, upper bound: 24.2570501
time: 33.32 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2874799, upper bound: 24.3243647
time: 53.13 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -22.2029076, 15.9588795, -22.2500648, 15.8886547, -33.7251129, 33.8301430
1: -1.9698646, 26.1093388, -2.0676403, 26.0752525, -23.7730942, 23.8995819
2: 1.4616554, 24.3190289, 1.4106627, 24.3108749, -20.4889221, 20.5607147
3: -12.9259682, 14.6865711, -12.9446173, 14.6378603, -23.5901985, 23.6911392
4: -1.8620870, 26.2381134, -1.9079564, 26.2335892, -24.3566055, 24.4387093
5: 0.2576375, 23.3729095, 0.2259359, 23.3396339, -20.7250824, 20.8273163
6: -74.5574036, -34.0568848, -74.5364075, -34.0140724, -29.2324181, 29.2429466
7: -5.0043583, 19.2788620, -5.0446129, 19.2588425, -20.0391464, 20.1230412
8: -8.9149227, 20.5848656, -8.9954309, 20.5729084, -28.0308647, 28.1712799
9: -26.2688732, 10.9849358, -26.3204746, 10.9149446, -30.6057663, 30.7455254
10: -35.4057579, 11.0019970, -35.4162750, 10.9230804, -37.1463547, 37.2643013
11: -25.0396500, 9.9610252, -25.0078354, 9.9621954, -27.6181717, 27.5468788
12: -42.5859070, -5.9156919, -42.5734329, -5.9284120, -26.1478348, 26.1942062
13: -38.5561905, 20.2634182, -38.5486603, 20.3042526, -43.4649963, 43.3838844
14: 1.8307209, 49.1478615, 1.7750645, 49.0534897, -39.5173950, 39.6912384
15: -16.9975586, 19.4955750, -17.0326481, 19.4686680, -31.8995056, 31.9856033
16: -38.0934525, 6.7475581, -38.1810646, 6.6791096, -34.2495918, 34.4373016
17: 9.8849831, 55.1301117, 9.8957834, 55.1099854, -36.0633430, 36.1053009
18: -8.1971846, 38.7563782, -8.2352467, 38.7458267, -41.3240204, 41.4487305
19: -18.2249832, 17.4116402, -18.1251984, 17.4734516, -30.8619385, 30.5832748
20: -16.2126274, 18.7501469, -16.1324005, 18.7958107, -32.0266800, 31.8821144
21: -16.1294670, 19.3171101, -16.0741959, 19.3708172, -33.5921631, 33.4067917
22: -2.4518914, 31.4833794, -2.4065304, 31.5719547, -27.7423630, 27.5913010
23: -14.9930115, 15.3213234, -14.9083290, 15.3462763, -28.8269348, 28.6075668
24: -8.8797312, 29.2769184, -8.8146887, 29.3077698, -33.6369095, 33.4346924
25: 1.1540051, 39.0193710, 1.1941719, 39.0803185, -34.4168930, 34.1590309
26: -16.3260326, 24.3496323, -16.2608452, 24.3571892, -36.8030396, 36.6469460
27: -14.4584141, 21.7384396, -14.4291553, 21.7418041, -26.8444557, 26.8282146
28: -5.5702543, 23.9280853, -5.4996700, 23.9657784, -29.3610916, 29.1786079
29: -6.3352633, 24.2101631, -6.3164263, 24.2808781, -26.2364426, 26.1250401
30: -12.5541973, 22.1162910, -12.5283842, 22.1717281, -30.5072212, 30.4204941
31: -17.2998028, 27.2493210, -17.2438335, 27.2811127, -39.4970856, 39.2312393
32: -70.1836548, -28.9457874, -70.1725159, -28.8812523, -30.8132172, 30.7714691
33: -110.4006348, -44.7854576, -110.3285599, -44.6183586, -45.5863495, 45.3073273
34: -88.7594376, -42.1680145, -88.7543945, -42.0536385, -29.1584778, 29.0554314
35: -66.5845795, -20.9352455, -66.5621338, -20.8291149, -30.9641113, 30.8089867
36: -60.8809280, -16.0909882, -60.8209000, -15.9964323, -29.6976776, 29.5227814
37: -121.4903259, -55.2384415, -121.3680496, -55.1852493, -46.2734680, 46.0010376
38: -70.7466583, -15.6453876, -70.6699905, -15.5886517, -39.8665848, 39.7022095
39: -111.1480331, -44.4436493, -111.0537872, -44.3153496, -41.7075500, 41.4123154
40: -114.8668976, -64.8474579, -114.8265686, -64.8511887, -35.5617218, 35.6736870
41: -87.2472382, -42.5706635, -87.1799011, -42.5260010, -29.5861282, 29.5072956
42: -72.5902557, -35.9723206, -72.5804291, -35.9436035, -30.1668243, 30.1307678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3213425, upper bound: 24.1834914
time: 44.96 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3218920, upper bound: 24.2508041
time: 39.19 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -22.2046223, 15.9765577, -22.2798729, 15.9254084, -33.7447891, 33.8733215
1: -1.9707477, 26.1119308, -2.0751162, 26.0821800, -23.7781296, 23.9215736
2: 1.4601777, 24.3225021, 1.3998277, 24.3193092, -20.4967804, 20.5738144
3: -12.9326916, 14.6910753, -12.9611816, 14.6519651, -23.6108170, 23.7076626
4: -1.8647511, 26.2447453, -1.9441087, 26.2487679, -24.3732262, 24.4832306
5: 0.2529182, 23.3756981, 0.2122521, 23.3474884, -20.7319145, 20.8603859
6: -74.5645676, -34.0554085, -74.5527039, -33.9789505, -29.2787857, 29.2407875
7: -5.0123444, 19.2818832, -5.0620165, 19.2799454, -20.0718918, 20.1303902
8: -8.9170723, 20.5877037, -9.0119581, 20.5800228, -28.0604095, 28.1831512
9: -26.2701683, 10.9987440, -26.3343067, 10.9505978, -30.6290054, 30.7862549
10: -35.4077148, 11.0152006, -35.4286613, 10.9646988, -37.1729431, 37.3063660
11: -25.0463715, 9.9628754, -25.0236626, 10.0037994, -27.6771736, 27.5521202
12: -42.6004333, -5.9141874, -42.6052666, -5.8935671, -26.2023163, 26.2093620
13: -38.5709991, 20.2659378, -38.5795631, 20.3488770, -43.5172119, 43.3876953
14: 1.8124762, 49.1493073, 1.7274027, 49.0714264, -39.5521011, 39.7244797
15: -16.9994354, 19.5098915, -17.0981026, 19.5007019, -31.9065399, 32.0806923
16: -38.0944672, 6.7647409, -38.2171898, 6.7268643, -34.2663612, 34.4844017
17: 9.8511591, 55.1305313, 9.8193169, 55.1506691, -36.1480560, 36.1378403
18: -8.2016916, 38.7703285, -8.2847710, 38.7742882, -41.3514481, 41.5127182
19: -18.2284164, 17.4136505, -18.1450348, 17.4812469, -30.8842010, 30.5896034
20: -16.2249413, 18.7514420, -16.1588440, 18.8213215, -32.0686111, 31.8994827
21: -16.1336174, 19.3183346, -16.0874710, 19.3834324, -33.6393127, 33.4178619
22: -2.4557271, 31.4836674, -2.4292264, 31.5736523, -27.7499313, 27.6149559
23: -14.9954510, 15.3260326, -14.9198389, 15.3589096, -28.8841248, 28.6013107
24: -8.8824739, 29.2780266, -8.8325424, 29.3110619, -33.6412735, 33.4573059
25: 1.1476707, 39.0208511, 1.1735935, 39.0873108, -34.4522476, 34.1790314
26: -16.3330498, 24.3530807, -16.2922726, 24.3646336, -36.8231583, 36.6689072
27: -14.4598627, 21.7409306, -14.4510126, 21.7482681, -26.8483849, 26.8525925
28: -5.5734787, 23.9299488, -5.5099869, 23.9714050, -29.4047508, 29.1867065
29: -6.3389778, 24.2111702, -6.3273411, 24.2846336, -26.2451515, 26.1360054
30: -12.5709839, 22.1188812, -12.5655317, 22.2234764, -30.5785599, 30.4326477
31: -17.3046017, 27.2519741, -17.2646904, 27.2875977, -39.5052185, 39.2528076
32: -70.1892700, -28.9436226, -70.1854019, -28.8493595, -30.8563118, 30.7743225
33: -110.4040070, -44.7798080, -110.3455887, -44.6042633, -45.6126633, 45.3203964
34: -88.7605438, -42.1636238, -88.7652283, -42.0388947, -29.1665192, 29.0747681
35: -66.5872498, -20.9304733, -66.5777283, -20.8168793, -30.9790726, 30.8246498
36: -60.8838005, -16.0901947, -60.8294792, -15.9902420, -29.7073174, 29.5342331
37: -121.4964142, -55.2135468, -121.4226379, -55.1309967, -46.2859116, 46.0762634
38: -70.7549591, -15.6439085, -70.6941452, -15.5838995, -39.8757019, 39.7381592
39: -111.1507339, -44.4384041, -111.0690079, -44.3001671, -41.7315674, 41.4246597
40: -114.8697433, -64.8285522, -114.8638382, -64.8064575, -35.5718307, 35.7209778
41: -87.2477036, -42.5583801, -87.1996307, -42.4918365, -29.6058044, 29.5272427
42: -72.5913849, -35.9693718, -72.5843811, -35.9153175, -30.2274704, 30.1302910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3238254, upper bound: 24.2570499
time: 27.84 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3243647, upper bound: 24.3243645
time: 44.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 75.14 seconds
IS_A1_B1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2082007, upper bound: 24.2570276
IS_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2087457, upper bound: 24.3243366
IS_A1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2509244, upper bound: 24.2570501
IS_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2514694, upper bound: 24.3243647
IS_A1_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.1705793, upper bound: 24.2545369
IS_A1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.1711362, upper bound: 24.3218225
IS_A1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2441925, upper bound: 24.2570276
IS_A1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2447324, upper bound: 24.3243366
IS_A1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2133305, upper bound: 24.2545968
IS_A1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2138847, upper bound: 24.3218920
IS_A1_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2869402, upper bound: 24.2570501
IS_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.2874799, upper bound: 24.3243647
IS_A2_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.3213425, upper bound: 24.1834914
IS_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.3218920, upper bound: 24.2508041
IS_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.3238254, upper bound: 24.2570499
IS_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 75.14
Output dim: 17, lower bound: -24.3243647, upper bound: 24.3243645

## BFS IS instance: IS_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -22.0979004, 15.8625793, -22.1264782, 15.9092636, -33.7038040, 33.5894547
1: -1.9222043, 26.0242405, -1.9209764, 26.0815697, -23.7442856, 23.6735764
2: 1.4845078, 24.2594204, 1.5014551, 24.2984924, -20.4385147, 20.4026909
3: -12.8918190, 14.5973368, -12.9117432, 14.6368313, -23.5527840, 23.5838928
4: -1.8254828, 26.1709824, -1.7804530, 26.2135773, -24.3263779, 24.2224388
5: 0.2918582, 23.3061981, 0.2772584, 23.3333359, -20.6791954, 20.6971321
6: -74.4606171, -34.1004105, -74.5469971, -34.1307068, -29.0391769, 29.2057877
7: -4.9989452, 19.2332764, -4.9965096, 19.2654190, -20.0274391, 20.0200996
8: -8.8525658, 20.4915047, -8.8317375, 20.5432358, -27.9375343, 27.9100838
9: -26.2119827, 10.8608131, -26.2371750, 10.8992128, -30.5339432, 30.5142632
10: -35.3423157, 10.8774490, -35.3750305, 10.8746290, -37.0478668, 36.9784775
11: -24.9813843, 9.9026394, -25.0030251, 9.8772402, -27.4438248, 27.4674454
12: -42.4561081, -6.0118408, -42.5248795, -6.0580111, -25.8903160, 25.9904518
13: -38.4416771, 20.1376972, -38.4923973, 20.1163406, -43.1514816, 43.2374535
14: 1.9441710, 49.0041428, 1.9055214, 49.0449333, -39.3840332, 39.4061356
15: -16.9363174, 19.4204693, -16.9036217, 19.4378757, -31.8477859, 31.7380028
16: -38.0410042, 6.6445756, -38.0577850, 6.6923103, -34.1852989, 34.1808853
17: 10.0691185, 55.0265045, 9.9806080, 55.0478821, -35.7918167, 35.8060837
18: -8.1295853, 38.7018204, -8.0922585, 38.7427330, -41.4154663, 41.1987991
19: -18.1052589, 17.3485184, -18.1350975, 17.3798485, -30.6231918, 30.5272865
20: -16.1057510, 18.6623631, -16.1528358, 18.6777859, -31.7692032, 31.8232384
21: -16.0295162, 19.2439919, -16.0597630, 19.2570953, -33.3286438, 33.3480911
22: -2.3561602, 31.4551125, -2.3856854, 31.4685402, -27.5413895, 27.5531731
23: -14.8780708, 15.2520447, -14.9155064, 15.2826338, -28.5584717, 28.5490189
24: -8.7741451, 29.2140675, -8.7989664, 29.2419243, -33.4686432, 33.3791428
25: 1.2370567, 38.9617653, 1.2072582, 38.9804077, -34.1668015, 34.1249580
26: -16.2195168, 24.2802639, -16.2403488, 24.3258781, -36.6028519, 36.5656929
27: -14.3159552, 21.6317749, -14.3027496, 21.6826935, -26.5180664, 26.5986233
28: -5.4669027, 23.8607635, -5.4831533, 23.8945999, -29.1093216, 29.1069527
29: -6.2491407, 24.1794777, -6.2928429, 24.1747665, -26.0397110, 26.0883255
30: -12.4731197, 22.0443535, -12.5099688, 22.0195751, -30.2529755, 30.3817825
31: -17.1962166, 27.1965504, -17.2349052, 27.2213459, -39.3419342, 39.1989212
32: -70.1053162, -28.9798660, -70.1849365, -29.0026855, -30.5938225, 30.7808762
33: -110.2080002, -44.8464813, -110.3228302, -44.8407669, -45.1788864, 45.2762604
34: -88.6753540, -42.1879578, -88.7593536, -42.2016106, -28.9144554, 29.0300522
35: -66.4835587, -20.9605007, -66.5591583, -20.9644127, -30.7223740, 30.7909203
36: -60.7560425, -16.1416130, -60.8264198, -16.1213417, -29.4242020, 29.4305267
37: -121.2684479, -55.2901688, -121.3773651, -55.2786064, -45.9594116, 45.9880676
38: -70.6051407, -15.7070770, -70.6847916, -15.6829920, -39.6664581, 39.6623306
39: -110.9517517, -44.5114365, -111.0557327, -44.5016975, -41.3220520, 41.3763351
40: -114.7782974, -64.8637466, -114.8302002, -64.8724823, -35.6062164, 35.4890900
41: -87.1170883, -42.6215401, -87.1952667, -42.6085587, -29.3098793, 29.4400063
42: -72.5117035, -36.0290642, -72.5794907, -36.0481682, -29.9691696, 30.0843887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1999949, upper bound: 24.2313472
time: 41.38 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2073967, upper bound: 24.3229815
time: 43.09 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.1931610, 15.9211731, -22.1301899, 15.9262810, -33.8176956, 33.6517944
1: -2.0315440, 26.0806274, -1.9220946, 26.0998802, -23.8729248, 23.7270546
2: 1.4208672, 24.3124104, 1.5005672, 24.3144951, -20.5199013, 20.4564877
3: -12.9261971, 14.6437588, -12.9132032, 14.6491909, -23.6030121, 23.6328926
4: -1.9072988, 26.2338142, -1.7820084, 26.2323990, -24.4315605, 24.2847710
5: 0.2414994, 23.3431854, 0.2761359, 23.3430710, -20.7445984, 20.7347031
6: -74.5442581, -34.0246887, -74.5677338, -34.1304703, -29.1161804, 29.2884407
7: -5.0585775, 19.2740135, -4.9980316, 19.2767563, -20.1003380, 20.0627689
8: -8.9907665, 20.5480118, -8.8329763, 20.5624657, -28.1051903, 27.9655151
9: -26.2847290, 10.9339180, -26.2379761, 10.9188852, -30.6378021, 30.5875473
10: -35.3810959, 10.9364929, -35.3771439, 10.8846397, -37.1338196, 37.0415688
11: -25.0142193, 9.9373713, -25.0061188, 9.8780937, -27.4828491, 27.5413818
12: -42.5455093, -5.9271574, -42.5546532, -6.0571513, -25.9687042, 26.1109200
13: -38.5247421, 20.3021965, -38.5189476, 20.1203995, -43.2262878, 43.4374542
14: 1.8181086, 49.0491028, 1.8969030, 49.0572968, -39.5185547, 39.4575119
15: -17.0368671, 19.4917812, -16.9047680, 19.4554634, -31.9867783, 31.8050308
16: -38.1784668, 6.7125931, -38.0592422, 6.7111607, -34.3563309, 34.2443314
17: 9.9181519, 55.1218491, 9.9404640, 55.0481491, -35.9350052, 35.9475479
18: -8.2309322, 38.7722969, -8.0965500, 38.7664604, -41.5450897, 41.2696228
19: -18.1446304, 17.4176846, -18.1415520, 17.3806534, -30.6584854, 30.6386986
20: -16.1550045, 18.7400513, -16.1658974, 18.6800232, -31.8182755, 31.9318390
21: -16.0823097, 19.3133526, -16.0670700, 19.2591400, -33.3685303, 33.4797440
22: -2.4177208, 31.5492821, -2.4024086, 31.4702511, -27.6035919, 27.6705017
23: -14.9113483, 15.2971325, -14.9207439, 15.2840452, -28.5821457, 28.6597748
24: -8.8073168, 29.2522774, -8.8015242, 29.2436733, -33.4989471, 33.4678307
25: 1.1847925, 39.0352936, 1.2016277, 38.9831543, -34.2051468, 34.2824287
26: -16.2847366, 24.3067608, -16.2474117, 24.3280964, -36.6515656, 36.6488724
27: -14.4251146, 21.6820564, -14.3063507, 21.6983337, -26.6518593, 26.6473274
28: -5.5040207, 23.9167976, -5.4895544, 23.8969688, -29.1427727, 29.2201691
29: -6.3190918, 24.2542076, -6.3112507, 24.1762352, -26.1101151, 26.1845913
30: -12.5576210, 22.1531181, -12.5289106, 22.0217762, -30.3337898, 30.5119324
31: -17.2480755, 27.2354813, -17.2410736, 27.2224979, -39.3784180, 39.3121185
32: -70.1760483, -28.8744049, -70.2055511, -29.0007610, -30.6646690, 30.9115963
33: -110.3173370, -44.6594086, -110.3589783, -44.8387146, -45.2665405, 45.5244141
34: -88.7456512, -42.0541344, -88.7803040, -42.2008362, -28.9759598, 29.1869011
35: -66.5563278, -20.8400955, -66.5825424, -20.9639225, -30.7843781, 30.9483299
36: -60.8207893, -16.0359993, -60.8473473, -16.1206131, -29.4829559, 29.5625343
37: -121.3807983, -55.2012329, -121.4113007, -55.2778664, -46.0615234, 46.1352005
38: -70.6771851, -15.6419344, -70.7034454, -15.6824207, -39.7356415, 39.7616272
39: -111.0473328, -44.3684616, -111.0865631, -44.5008774, -41.4036713, 41.5670013
40: -114.8338623, -64.8343811, -114.8378220, -64.8724136, -35.6984329, 35.5115204
41: -87.1780090, -42.5495605, -87.2070770, -42.6074028, -29.3801689, 29.5180397
42: -72.5760040, -35.9687767, -72.5907593, -36.0470200, -30.0299225, 30.1574097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2427196, upper bound: 24.2313559
time: 33.65 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2501203, upper bound: 24.3230109
time: 39.82 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -22.0870476, 15.8267097, -22.1710930, 15.9121170, -33.6752090, 33.6267929
1: -1.9292529, 26.0157719, -1.9508996, 26.0937595, -23.7489090, 23.6919899
2: 1.4801707, 24.2508354, 1.4699962, 24.3121681, -20.4562340, 20.4165573
3: -12.8769350, 14.5793190, -12.9111481, 14.6501255, -23.5781097, 23.5481758
4: -1.8203771, 26.1568413, -1.8515835, 26.2289162, -24.3340454, 24.2670670
5: 0.3051505, 23.2973251, 0.2746425, 23.3401070, -20.7107964, 20.6587563
6: -74.4426651, -34.1113663, -74.5619202, -34.0748940, -29.1249695, 29.1642494
7: -4.9803901, 19.2123413, -4.9982710, 19.2768116, -20.0456581, 19.9856224
8: -8.8518057, 20.4845982, -8.8690586, 20.5550461, -27.9519806, 27.9004478
9: -26.1958733, 10.8293438, -26.2451439, 10.9085340, -30.5359306, 30.4846573
10: -35.3251266, 10.8494720, -35.3939209, 10.9087753, -37.0527878, 36.9685783
11: -24.9648609, 9.8927002, -25.0170326, 9.9422083, -27.4793549, 27.4745674
12: -42.4253654, -6.0168853, -42.5430145, -5.9874272, -25.9817810, 25.9420128
13: -38.4102478, 20.1268597, -38.5010757, 20.1931801, -43.1947861, 43.2509232
14: 1.9861231, 49.0040359, 1.8781366, 49.0847092, -39.4070511, 39.4101410
15: -16.9072800, 19.3866692, -16.9768066, 19.4490108, -31.7990723, 31.8054276
16: -37.9991379, 6.6058140, -38.0705872, 6.6931190, -34.1851807, 34.1633148
17: 10.1427698, 55.0107231, 9.9682417, 55.1010590, -35.7945213, 35.7811241
18: -8.1149311, 38.6732559, -8.1748848, 38.7537003, -41.4007187, 41.2394104
19: -18.0791359, 17.3420410, -18.1517029, 17.3826237, -30.5466766, 30.6152878
20: -16.0797081, 18.6550102, -16.1596069, 18.7155495, -31.7909317, 31.8143845
21: -16.0135269, 19.2455883, -16.0769348, 19.2854538, -33.3317184, 33.3594933
22: -2.3401136, 31.4526386, -2.4028206, 31.4668732, -27.5198822, 27.5762520
23: -14.8679028, 15.2401323, -14.9253798, 15.2859955, -28.5112991, 28.5926552
24: -8.7681904, 29.2112389, -8.8338223, 29.2507057, -33.4285355, 33.4483910
25: 1.2559414, 38.9615936, 1.1955338, 38.9934921, -34.1036987, 34.1772079
26: -16.1884079, 24.2721977, -16.2584705, 24.3221531, -36.5710144, 36.6018448
27: -14.3146219, 21.6249371, -14.3575745, 21.6938648, -26.5310822, 26.6369152
28: -5.4572687, 23.8560371, -5.4920483, 23.8979549, -29.0866013, 29.1265640
29: -6.2376223, 24.1879883, -6.3103905, 24.2005043, -26.0372047, 26.1175175
30: -12.4356699, 22.0266361, -12.5188942, 22.0881939, -30.3043938, 30.3575401
31: -17.1767521, 27.1901321, -17.2580299, 27.2229424, -39.2485962, 39.2920380
32: -70.0923309, -28.9982681, -70.1971207, -28.9604492, -30.6575394, 30.7365456
33: -110.2117157, -44.8599396, -110.3673859, -44.8172379, -45.1787262, 45.3061676
34: -88.6793823, -42.2047005, -88.7979584, -42.1800919, -28.9556541, 29.0311775
35: -66.4804840, -20.9731483, -66.5888062, -20.9477577, -30.7173996, 30.8151321
36: -60.7491837, -16.1480904, -60.8374062, -16.1132755, -29.4308891, 29.4257317
37: -121.2359619, -55.3443756, -121.4329529, -55.2737274, -45.8991699, 46.0263138
38: -70.5878372, -15.7129974, -70.7040710, -15.6717224, -39.6519012, 39.6736221
39: -110.9508667, -44.5264053, -111.0920792, -44.4794235, -41.2739563, 41.4298706
40: -114.7617188, -64.9069366, -114.8748932, -64.8639832, -35.6395798, 35.4687271
41: -87.1103897, -42.6528778, -87.2260284, -42.5948067, -29.3444061, 29.4087524
42: -72.5087128, -36.0376129, -72.5929565, -35.9961967, -30.0155029, 30.0784454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1060368, upper bound: 24.3141376
time: 51.82 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1681453, upper bound: 24.3202323
time: 48.80 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -22.1168709, 15.8634453, -22.1728058, 15.9297867, -33.7183800, 33.6465111
1: -1.9367127, 26.0226936, -1.9517882, 26.0963688, -23.7708778, 23.6970215
2: 1.4693267, 24.2592621, 1.4685349, 24.3156338, -20.4693604, 20.4243965
3: -12.8935022, 14.5933886, -12.9178934, 14.6546335, -23.5946732, 23.5687866
4: -1.8565176, 26.1720047, -1.8542457, 26.2355843, -24.3785591, 24.2836838
5: 0.2914562, 23.3051777, 0.2699227, 23.3429070, -20.7438774, 20.6655846
6: -74.4589310, -34.0762444, -74.5691528, -34.0733795, -29.1228104, 29.2106247
7: -4.9977760, 19.2334328, -5.0062342, 19.2798309, -20.0530090, 20.0183544
8: -8.8682814, 20.4917126, -8.8712215, 20.5578403, -27.9638748, 27.9299965
9: -26.2096863, 10.8650331, -26.2464142, 10.9223518, -30.5766449, 30.5078812
10: -35.3374901, 10.8911419, -35.3958855, 10.9219961, -37.0948563, 36.9951973
11: -24.9806690, 9.9343014, -25.0237770, 9.9441156, -27.4846153, 27.5335922
12: -42.4571991, -5.9820580, -42.5575256, -5.9859262, -25.9969292, 25.9964218
13: -38.4411201, 20.1714363, -38.5158958, 20.1956825, -43.1985779, 43.3031006
14: 1.9383926, 49.0219383, 1.8598900, 49.0861168, -39.4402657, 39.4447021
15: -16.9727211, 19.4187622, -16.9787121, 19.4633331, -31.8941574, 31.8124962
16: -38.0352631, 6.6535625, -38.0716667, 6.7102489, -34.2323074, 34.1800385
17: 10.0663300, 55.0513992, 9.9344234, 55.1014709, -35.8270988, 35.8658333
18: -8.1644640, 38.7017517, -8.1794214, 38.7676544, -41.4647369, 41.2667847
19: -18.0988102, 17.3497963, -18.1551399, 17.3846359, -30.5529633, 30.6375351
20: -16.1062279, 18.6805019, -16.1719360, 18.7168617, -31.8082886, 31.8563042
21: -16.0267792, 19.2582073, -16.0810966, 19.2866402, -33.3427963, 33.4066849
22: -2.3628006, 31.4543495, -2.4066496, 31.4671459, -27.5434608, 27.5837784
23: -14.8793507, 15.2527418, -14.9278221, 15.2906885, -28.5050659, 28.6498222
24: -8.7860203, 29.2145386, -8.8365269, 29.2518082, -33.4511337, 33.4527702
25: 1.2353745, 38.9685860, 1.1892014, 38.9949417, -34.1237259, 34.2125778
26: -16.2197361, 24.2796555, -16.2654762, 24.3255863, -36.5928879, 36.6219711
27: -14.3364105, 21.6314087, -14.3590393, 21.6964111, -26.5554276, 26.6408615
28: -5.4675550, 23.8616638, -5.4953003, 23.8998203, -29.0947495, 29.1702118
29: -6.2485647, 24.1917744, -6.3141079, 24.2015038, -26.0481567, 26.1262589
30: -12.4728098, 22.0784016, -12.5356426, 22.0907784, -30.3166275, 30.4288864
31: -17.1975174, 27.1966305, -17.2628479, 27.2256184, -39.2700500, 39.3001251
32: -70.1051712, -28.9663754, -70.2027283, -28.9582710, -30.6604233, 30.7796707
33: -110.2286224, -44.8459206, -110.3708115, -44.8115959, -45.1916351, 45.3323364
34: -88.6902618, -42.1898842, -88.7990417, -42.1757660, -28.9750023, 29.0392532
35: -66.4960327, -20.9608879, -66.5914917, -20.9429874, -30.7330780, 30.8300743
36: -60.7577667, -16.1419086, -60.8403168, -16.1125431, -29.4423714, 29.4354095
37: -121.2905731, -55.2901001, -121.4390335, -55.2488632, -45.9743958, 46.0387192
38: -70.6119537, -15.7081909, -70.7123413, -15.6703186, -39.6878433, 39.6827164
39: -110.9661026, -44.5112190, -111.0947418, -44.4740372, -41.2862473, 41.4538422
40: -114.7990112, -64.8622131, -114.8776932, -64.8450699, -35.6868820, 35.4788742
41: -87.1301575, -42.6187897, -87.2265015, -42.5824966, -29.3643417, 29.4283924
42: -72.5126877, -36.0093040, -72.5941315, -35.9931793, -30.0150452, 30.1391258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1516943, upper bound: 24.3154917
time: 44.93 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2433791, upper bound: 24.3229818
time: 29.13 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -22.1823235, 15.8853083, -22.1748238, 15.9291105, -33.7891006, 33.6891632
1: -2.0386143, 26.0721474, -1.9520185, 26.1120872, -23.8775444, 23.7454605
2: 1.4165299, 24.3037968, 1.4691052, 24.3281593, -20.5376167, 20.4703407
3: -12.9113245, 14.6257086, -12.9126272, 14.6624727, -23.6283493, 23.5971489
4: -1.9021733, 26.2196541, -1.8531148, 26.2477341, -24.4392357, 24.3293724
5: 0.2547884, 23.3342972, 0.2735271, 23.3498230, -20.7761841, 20.6963043
6: -74.5262909, -34.0356445, -74.5826569, -34.0747070, -29.2020035, 29.2469177
7: -5.0400209, 19.2530556, -4.9998016, 19.2881451, -20.1185646, 20.0283089
8: -8.9899845, 20.5410538, -8.8703041, 20.5743027, -28.1196213, 27.9559097
9: -26.2687073, 10.9024487, -26.2459755, 10.9281416, -30.6397858, 30.5579224
10: -35.3638458, 10.9085884, -35.3959961, 10.9186935, -37.1387405, 37.0317116
11: -24.9977226, 9.9274025, -25.0201359, 9.9430695, -27.5183754, 27.5485153
12: -42.5147705, -5.9321814, -42.5727921, -5.9865623, -26.0601768, 26.0624924
13: -38.4933777, 20.2913017, -38.5276413, 20.1972160, -43.2696533, 43.4509087
14: 1.8600693, 49.0489464, 1.8695030, 49.0970573, -39.5416336, 39.4614830
15: -17.0077839, 19.4579983, -16.9779911, 19.4665604, -31.9380417, 31.8724747
16: -38.1365967, 6.6738682, -38.0720406, 6.7119007, -34.3562164, 34.2267532
17: 9.9917583, 55.1060181, 9.9280977, 55.1013908, -35.9376450, 35.9225578
18: -8.2163258, 38.7437439, -8.1792049, 38.7774353, -41.5303192, 41.3102493
19: -18.1183872, 17.4112034, -18.1581936, 17.3833866, -30.5819626, 30.7266846
20: -16.1289864, 18.7327023, -16.1726589, 18.7177868, -31.8399887, 31.9230118
21: -16.0663357, 19.3149662, -16.0842552, 19.2875385, -33.3715515, 33.4911423
22: -2.4016862, 31.5467987, -2.4195204, 31.4685802, -27.5821075, 27.6935768
23: -14.9011259, 15.2852192, -14.9306450, 15.2874107, -28.5349503, 28.7034073
24: -8.8013363, 29.2494564, -8.8363676, 29.2524624, -33.4588013, 33.5370407
25: 1.2036953, 39.0351181, 1.1898785, 38.9961967, -34.1421127, 34.3346558
26: -16.2535973, 24.2987099, -16.2655067, 24.3243656, -36.6197433, 36.6850433
27: -14.4237423, 21.6752396, -14.3611774, 21.7095394, -26.6648636, 26.6856079
28: -5.4943352, 23.9120541, -5.4984121, 23.9003220, -29.1200638, 29.2397842
29: -6.3075838, 24.2627335, -6.3288221, 24.2019806, -26.1075974, 26.2137642
30: -12.5201969, 22.1354008, -12.5378151, 22.0904160, -30.3852463, 30.4876556
31: -17.2285442, 27.2290916, -17.2642212, 27.2240829, -39.2850342, 39.4052353
32: -70.1629791, -28.8927956, -70.2177124, -28.9585228, -30.7283859, 30.8672752
33: -110.3210220, -44.6728058, -110.4035645, -44.8152466, -45.2664642, 45.5542984
34: -88.7497406, -42.0708313, -88.8188782, -42.1793365, -29.0171661, 29.1880035
35: -66.5532990, -20.8526745, -66.6121902, -20.9473152, -30.7794189, 30.9724770
36: -60.8139229, -16.0425301, -60.8583717, -16.1125832, -29.4896507, 29.5577698
37: -121.3481979, -55.2554398, -121.4669418, -55.2729492, -46.0013504, 46.1734085
38: -70.6599121, -15.6478910, -70.7227631, -15.6711025, -39.7209854, 39.7728577
39: -111.0464630, -44.3834953, -111.1229553, -44.4786224, -41.3555603, 41.6205063
40: -114.8172684, -64.8775635, -114.8825531, -64.8639297, -35.7318039, 35.4911575
41: -87.1712646, -42.5809250, -87.2378387, -42.5936699, -29.4146805, 29.4867630
42: -72.5729980, -35.9773331, -72.6042175, -35.9950790, -30.0762711, 30.1514969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2043580, upper bound: 24.2293169
time: 32.27 seconds

## Relational analysis of IS_A1_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2108955, upper bound: 24.3203017
time: 46.42 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -22.2121162, 15.9220362, -22.1765137, 15.9468174, -33.8322830, 33.7088547
1: -2.0460608, 26.0790806, -1.9528973, 26.1146889, -23.8995285, 23.7504883
2: 1.4056978, 24.3122540, 1.4676404, 24.3316326, -20.5507278, 20.4782124
3: -12.9278889, 14.6398163, -12.9193420, 14.6669884, -23.6449127, 23.6177750
4: -1.9383256, 26.2348175, -1.8557901, 26.2543945, -24.4837570, 24.3460159
5: 0.2411065, 23.3421440, 0.2688112, 23.3526363, -20.8092613, 20.7031288
6: -74.5425873, -34.0005112, -74.5898743, -34.0732117, -29.1998291, 29.2932434
7: -5.0574446, 19.2741508, -5.0077724, 19.2911701, -20.1259117, 20.0610580
8: -9.0065269, 20.5482063, -8.8724585, 20.5771103, -28.1315231, 27.9854584
9: -26.2824421, 10.9381485, -26.2472076, 10.9419823, -30.6805077, 30.5811539
10: -35.3762703, 10.9502354, -35.3979568, 10.9319620, -37.1808701, 37.0583420
11: -25.0135384, 9.9690390, -25.0268936, 9.9449501, -27.5235863, 27.6075439
12: -42.5466080, -5.8973746, -42.5873146, -5.9850483, -26.0752831, 26.1169090
13: -38.5242462, 20.3360062, -38.5423393, 20.1997166, -43.2734222, 43.5031395
14: 1.8123436, 49.0668716, 1.8512583, 49.0984612, -39.5748444, 39.4961090
15: -17.0732822, 19.4900513, -16.9798965, 19.4808846, -32.0331573, 31.8795166
16: -38.1727409, 6.7215366, -38.0730705, 6.7290640, -34.4033051, 34.2435303
17: 9.9153328, 55.1467094, 9.8942938, 55.1018295, -35.9702148, 36.0072365
18: -8.2658129, 38.7722244, -8.1837626, 38.7913437, -41.5943298, 41.3376236
19: -18.1381702, 17.4189911, -18.1615829, 17.3854122, -30.5882645, 30.7489395
20: -16.1554375, 18.7582169, -16.1849747, 18.7190781, -31.8573151, 31.9649506
21: -16.0795746, 19.3275642, -16.0883980, 19.2887001, -33.3826599, 33.5383110
22: -2.4243703, 31.5485115, -2.4233637, 31.4688435, -27.6057281, 27.7011070
23: -14.9126511, 15.2978668, -14.9330721, 15.2921314, -28.5287628, 28.7605782
24: -8.8192253, 29.2527103, -8.8390932, 29.2535954, -33.4814606, 33.5414391
25: 1.1831169, 39.0420837, 1.1835594, 38.9977112, -34.1621246, 34.3700447
26: -16.2849998, 24.3061638, -16.2725353, 24.3278122, -36.6416473, 36.7052078
27: -14.4455881, 21.6816921, -14.3626566, 21.7121010, -26.6892471, 26.6895447
28: -5.5046349, 23.9176884, -5.5016727, 23.9021759, -29.1281815, 29.2834435
29: -6.3185215, 24.2665043, -6.3325167, 24.2029400, -26.1185684, 26.2225208
30: -12.5573168, 22.1871681, -12.5545654, 22.0929794, -30.3974228, 30.5590172
31: -17.2493515, 27.2355537, -17.2690201, 27.2267590, -39.3065948, 39.4133377
32: -70.1758652, -28.8608665, -70.2233429, -28.9563560, -30.7312775, 30.9103642
33: -110.3379974, -44.6586952, -110.4070129, -44.8095360, -45.2794647, 45.5805283
34: -88.7605286, -42.0560875, -88.8199463, -42.1750221, -29.0365143, 29.1960907
35: -66.5688477, -20.8404579, -66.6149063, -20.9424534, -30.7951355, 30.9874611
36: -60.8225479, -16.0362663, -60.8612900, -16.1117401, -29.5011520, 29.5674248
37: -121.4028702, -55.2012215, -121.4729385, -55.2480698, -46.0766144, 46.1859360
38: -70.6839523, -15.6431255, -70.7310181, -15.6696825, -39.7569885, 39.7819595
39: -111.0616455, -44.3683243, -111.1255951, -44.4733620, -41.3678970, 41.6444931
40: -114.8545837, -64.8327942, -114.8853149, -64.8450012, -35.7790833, 35.5012589
41: -87.1910706, -42.5467644, -87.2383270, -42.5813293, -29.4346313, 29.5064449
42: -72.5769806, -35.9490471, -72.6053925, -35.9921036, -30.0758057, 30.2121582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2786409, upper bound: 24.2313559
time: 36.59 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.2861269, upper bound: 24.3230109
time: 53.05 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -22.2006569, 15.9583454, -22.2617416, 15.9222555, -33.7567863, 33.8319550
1: -1.9673369, 26.1089134, -2.0670600, 26.1126328, -23.8089790, 23.8892365
2: 1.4634914, 24.3186150, 1.4115279, 24.3333893, -20.5024452, 20.5563965
3: -12.9255333, 14.6860180, -12.9452915, 14.6616173, -23.6111526, 23.6857491
4: -1.8601751, 26.2375240, -1.9067440, 26.2555885, -24.3828735, 24.4284286
5: 0.2589927, 23.3724747, 0.2259984, 23.3546906, -20.7412872, 20.8232212
6: -74.5556259, -34.0607834, -74.5965118, -34.0195541, -29.2098694, 29.3056831
7: -5.0024920, 19.2785149, -5.0429435, 19.2786083, -20.0469208, 20.1165466
8: -8.9128494, 20.5843124, -8.9951887, 20.5852375, -28.0329742, 28.1746483
9: -26.2670517, 10.9841576, -26.3192844, 10.9291306, -30.6226654, 30.7385674
10: -35.4050293, 10.9995308, -35.4311028, 10.9231939, -37.1561966, 37.2397842
11: -25.0388546, 9.9593182, -25.0277863, 9.9602699, -27.6119728, 27.5627785
12: -42.5853195, -5.9170527, -42.5859413, -5.9284444, -26.1435394, 26.2081184
13: -38.5519524, 20.2616806, -38.5428772, 20.3211136, -43.4885559, 43.3784561
14: 1.8322105, 49.1470261, 1.7651720, 49.0989304, -39.5516701, 39.7080231
15: -16.9971676, 19.4938545, -17.0343819, 19.4714642, -31.9122849, 31.9729080
16: -38.0909004, 6.7472148, -38.1828194, 6.6969872, -34.2673912, 34.4385300
17: 9.8869543, 55.1294022, 9.8924704, 55.1643143, -36.0796890, 36.0899963
18: -8.1949310, 38.7561646, -8.2391529, 38.7690277, -41.3475952, 41.4457321
19: -18.2243042, 17.4087505, -18.1536674, 17.4684429, -30.8513794, 30.6043777
20: -16.2119141, 18.7475433, -16.1716881, 18.7955246, -32.0192261, 31.9191551
21: -16.1286049, 19.3145905, -16.1030502, 19.3668900, -33.5841980, 33.4342880
22: -2.4514203, 31.4805450, -2.4197531, 31.5673332, -27.7368393, 27.6110649
23: -14.9924192, 15.3189468, -14.9389782, 15.3428364, -28.8198166, 28.6368828
24: -8.8787470, 29.2746086, -8.8466120, 29.3051529, -33.6331100, 33.4682503
25: 1.1547613, 39.0172424, 1.1732826, 39.0776100, -34.4126434, 34.1813240
26: -16.3254013, 24.3478012, -16.2860241, 24.3590984, -36.8029938, 36.6708679
27: -14.4575186, 21.7363415, -14.4451447, 21.7389946, -26.8340073, 26.8499546
28: -5.5696583, 23.9267082, -5.5082216, 23.9667015, -29.3605118, 29.1912003
29: -6.3345985, 24.2088394, -6.3412390, 24.2799797, -26.2298660, 26.1491070
30: -12.5531263, 22.1144543, -12.5499477, 22.1701202, -30.5009689, 30.4400711
31: -17.2985458, 27.2454720, -17.2813339, 27.2743053, -39.4854736, 39.2620087
32: -70.1824341, -28.9491749, -70.2214737, -28.8839073, -30.7921829, 30.8241577
33: -110.3997803, -44.7910881, -110.4045258, -44.6252136, -45.5524445, 45.3984375
34: -88.7587433, -42.1727524, -88.8244324, -42.0590553, -29.1400452, 29.1025963
35: -66.5840530, -20.9385872, -66.6116180, -20.8310471, -30.9501724, 30.8637505
36: -60.8805008, -16.0946808, -60.8642578, -16.0003643, -29.6922455, 29.4930115
37: -121.4891663, -55.2440338, -121.4475021, -55.1924629, -46.2444458, 46.0681458
38: -70.7461548, -15.6492672, -70.7281952, -15.5922089, -39.8679047, 39.7333221
39: -111.1472473, -44.4492722, -111.1219406, -44.3243713, -41.6667786, 41.4810028
40: -114.8656921, -64.8503723, -114.8860931, -64.8526535, -35.5727768, 35.6330795
41: -87.2460403, -42.5747757, -87.2469711, -42.5269928, -29.5781326, 29.5438442
42: -72.5891418, -35.9765701, -72.6356583, -35.9482002, -30.1392212, 30.1954155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2293170, upper bound: 24.2412777
time: 44.30 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3203018, upper bound: 24.2478148
time: 58.24 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -22.1944008, 15.9747734, -22.2582703, 15.9216995, -33.7284393, 33.8460503
1: -1.9573770, 26.1110077, -2.0482240, 26.0802364, -23.7632370, 23.8944359
2: 1.4696825, 24.3207397, 1.4194915, 24.3156376, -20.4842796, 20.5560551
3: -12.9257078, 14.6893539, -12.9466686, 14.6483164, -23.6000862, 23.6912193
4: -1.8537109, 26.2429008, -1.9211698, 26.2448959, -24.3570442, 24.4558754
5: 0.2599034, 23.3743935, 0.2269654, 23.3447056, -20.7214928, 20.8424301
6: -74.5606689, -34.0768127, -74.5446625, -34.0241890, -29.2362671, 29.2121964
7: -5.0034571, 19.2807941, -5.0437856, 19.2776527, -20.0618896, 20.1145821
8: -8.9125080, 20.5864868, -9.0023613, 20.5774498, -28.0550690, 28.1758614
9: -26.2613049, 10.9965391, -26.3156395, 10.9461050, -30.6160774, 30.7659683
10: -35.4061432, 11.0061483, -35.4254189, 10.9456854, -37.1555099, 37.2885284
11: -25.0442162, 9.9554253, -25.0191574, 9.9880848, -27.6600304, 27.5408249
12: -42.5984001, -5.9185338, -42.6010551, -5.9027476, -26.1898956, 26.2002563
13: -38.5649223, 20.2611809, -38.5668755, 20.3390598, -43.5005798, 43.3671265
14: 1.8315258, 49.1476173, 1.7675085, 49.0680237, -39.5380630, 39.7037048
15: -16.9984913, 19.5069733, -17.0961685, 19.4946289, -31.8951797, 32.0673637
16: -38.0868378, 6.7636127, -38.2015800, 6.7244959, -34.2557220, 34.4660378
17: 9.8760710, 55.1293449, 9.8718882, 55.1481705, -36.1204567, 36.0913734
18: -8.1935081, 38.7694893, -8.2675266, 38.7725563, -41.3407593, 41.4937973
19: -18.2264938, 17.4029846, -18.1410255, 17.4587574, -30.8622971, 30.5763855
20: -16.2234249, 18.7376862, -16.1557274, 18.7922783, -32.0387421, 31.8831749
21: -16.1310081, 19.3075714, -16.0820694, 19.3610001, -33.6152496, 33.4029999
22: -2.4535065, 31.4778900, -2.4246540, 31.5616760, -27.7336121, 27.6036835
23: -14.9936409, 15.3158083, -14.9161215, 15.3373480, -28.8603516, 28.5874977
24: -8.8796215, 29.2743874, -8.8265553, 29.3034840, -33.6293945, 33.4479294
25: 1.1500835, 39.0143166, 1.1786060, 39.0738678, -34.4361267, 34.1682091
26: -16.3307514, 24.3479424, -16.2874908, 24.3538990, -36.8102722, 36.6595688
27: -14.4564257, 21.7356453, -14.4439278, 21.7375774, -26.8310471, 26.8386040
28: -5.5712662, 23.9227600, -5.5053949, 23.9562569, -29.3871918, 29.1751175
29: -6.3357334, 24.2040081, -6.3206043, 24.2695446, -26.2259216, 26.1213856
30: -12.5680561, 22.1110039, -12.5595045, 22.2068806, -30.5591431, 30.4189682
31: -17.3013687, 27.2430458, -17.2579365, 27.2687435, -39.4850388, 39.2388840
32: -70.1861038, -28.9614792, -70.1788101, -28.8868217, -30.8151817, 30.7495308
33: -110.4022064, -44.8110542, -110.3418121, -44.6703110, -45.5399933, 45.2838058
34: -88.7584457, -42.1899223, -88.7608337, -42.0943527, -29.1215210, 29.0478821
35: -66.5856018, -20.9476128, -66.5743256, -20.8528385, -30.9475708, 30.8061523
36: -60.8815384, -16.1092644, -60.8247986, -16.0304451, -29.6762695, 29.5089340
37: -121.4936905, -55.2445908, -121.4170074, -55.1962433, -46.2300873, 46.0424767
38: -70.7528915, -15.6654186, -70.6897888, -15.6286964, -39.8501892, 39.7212143
39: -111.1481857, -44.4664536, -111.0635834, -44.3558388, -41.6673737, 41.3880005
40: -114.8681335, -64.8524323, -114.8604431, -64.8563843, -35.5582275, 35.7121582
41: -87.2446823, -42.5821915, -87.1934052, -42.5421906, -29.5684891, 29.5009899
42: -72.5889740, -35.9930267, -72.5794144, -35.9653511, -30.1703949, 30.0995636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2308125, upper bound: 24.2482067
time: 52.54 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3224716, upper bound: 24.2556965
time: 51.08 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -22.2023907, 15.9760389, -22.2915497, 15.9590044, -33.7764893, 33.8751564
1: -1.9682519, 26.1115074, -2.0745180, 26.1195316, -23.8140297, 23.9112473
2: 1.4620259, 24.3220901, 1.4006865, 24.3418198, -20.5103416, 20.5694828
3: -12.9322424, 14.6905060, -12.9618635, 14.6756725, -23.6317444, 23.7022896
4: -1.8628266, 26.2441978, -1.9429018, 26.2707500, -24.3994675, 24.4729691
5: 0.2542820, 23.3752747, 0.2123156, 23.3625546, -20.7481270, 20.8563004
6: -74.5628510, -34.0593033, -74.6127472, -33.9843903, -29.2562485, 29.3034897
7: -5.0104799, 19.2815285, -5.0603461, 19.2997112, -20.0796738, 20.1239033
8: -8.9150095, 20.5871582, -9.0116854, 20.5923519, -28.0625038, 28.1865616
9: -26.2683258, 10.9979992, -26.3330688, 10.9648209, -30.6458664, 30.7793198
10: -35.4069366, 11.0127535, -35.4435043, 10.9647951, -37.1828003, 37.2818565
11: -25.0455666, 9.9612007, -25.0436287, 10.0018883, -27.6709900, 27.5680008
12: -42.5998688, -5.9155703, -42.6177826, -5.8935928, -26.1979828, 26.2232666
13: -38.5667458, 20.2641716, -38.5737915, 20.3657646, -43.5407486, 43.3822479
14: 1.8139896, 49.1484642, 1.7175188, 49.1168594, -39.5863953, 39.7412491
15: -16.9990616, 19.5082111, -17.0998573, 19.5035496, -31.9193192, 32.0679932
16: -38.0919724, 6.7644238, -38.2189560, 6.7447128, -34.2841339, 34.4856224
17: 9.8531704, 55.1297989, 9.8160725, 55.2049789, -36.1643906, 36.1225357
18: -8.1994648, 38.7701187, -8.2886724, 38.7975349, -41.3749847, 41.5097275
19: -18.2277355, 17.4107380, -18.1734657, 17.4762211, -30.8736038, 30.6107254
20: -16.2242165, 18.7488499, -16.1981506, 18.8210335, -32.0611420, 31.9365349
21: -16.1327477, 19.3157997, -16.1163425, 19.3794994, -33.6313782, 33.4453850
22: -2.4552765, 31.4808407, -2.4424396, 31.5690422, -27.7443810, 27.6347313
23: -14.9948502, 15.3236580, -14.9504967, 15.3554544, -28.8769684, 28.6306496
24: -8.8814487, 29.2757187, -8.8644695, 29.3084660, -33.6374893, 33.4909134
25: 1.1484432, 39.0187302, 1.1526871, 39.0846100, -34.4480286, 34.2013588
26: -16.3324013, 24.3512764, -16.3174324, 24.3665733, -36.8231201, 36.6928062
27: -14.4589844, 21.7388630, -14.4669752, 21.7454758, -26.8379021, 26.8743191
28: -5.5729122, 23.9285736, -5.5185146, 23.9723206, -29.4041634, 29.1992950
29: -6.3383398, 24.2098160, -6.3521643, 24.2837601, -26.2386017, 26.1600437
30: -12.5698977, 22.1170616, -12.5870562, 22.2218819, -30.5723190, 30.4522552
31: -17.3033676, 27.2481270, -17.3021183, 27.2808132, -39.4936066, 39.2835922
32: -70.1880341, -28.9470100, -70.2342834, -28.8520126, -30.8352852, 30.8270149
33: -110.4031677, -44.7854042, -110.4215240, -44.6110916, -45.5787506, 45.4114380
34: -88.7598648, -42.1684189, -88.8352356, -42.0443268, -29.1481438, 29.1219406
35: -66.5867615, -20.9337387, -66.6271515, -20.8187981, -30.9651184, 30.8794327
36: -60.8834038, -16.0939178, -60.8728561, -15.9941940, -29.7019043, 29.5044785
37: -121.4952393, -55.2191238, -121.5021439, -55.1382446, -46.2568512, 46.1434097
38: -70.7543793, -15.6478767, -70.7522125, -15.5874577, -39.8769836, 39.7692413
39: -111.1499176, -44.4440422, -111.1371384, -44.3091850, -41.6907806, 41.4933090
40: -114.8685837, -64.8314590, -114.9234467, -64.8079453, -35.5829010, 35.6803551
41: -87.2465210, -42.5624695, -87.2667160, -42.4928589, -29.5978470, 29.5637932
42: -72.5903320, -35.9735641, -72.6395874, -35.9199371, -30.1999054, 30.1949615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2313560, upper bound: 24.3155217
time: 42.63 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 17, lower bound: -24.3230109, upper bound: 24.3230109
time: 34.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 79.31 seconds
IS_A1_B1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.1999949, upper bound: 24.2313472
IS_A1_B1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2073967, upper bound: 24.3229815
IS_A1_B1_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2427196, upper bound: 24.2313559
IS_A1_B1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2501203, upper bound: 24.3230109
IS_A1_B2_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.1060368, upper bound: 24.3141376
IS_A1_B2_A2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.1681453, upper bound: 24.3202323
IS_A1_B2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.1516943, upper bound: 24.3154917
IS_A1_B2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2433791, upper bound: 24.3229818
IS_A1_B2_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2043580, upper bound: 24.2293169
IS_A1_B2_A2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2108955, upper bound: 24.3203017
IS_A1_B2_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2786409, upper bound: 24.2313559
IS_A1_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2861269, upper bound: 24.3230109
IS_A2_A2_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2293170, upper bound: 24.2412777
IS_A2_A2_B2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.3203018, upper bound: 24.2478148
IS_A2_A2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2308125, upper bound: 24.2482067
IS_A2_A2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.3224716, upper bound: 24.2556965
IS_A2_A2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.2313560, upper bound: 24.3155217
IS_A2_A2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.31
Output dim: 17, lower bound: -24.3230109, upper bound: 24.3230109

## BFS IS instance: IS_A1_B1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -22.0968323, 15.8617077, -22.1719398, 15.9095173, -33.6943283, 33.6250458
1: -1.9196455, 26.0235462, -1.9174798, 26.0866165, -23.7456551, 23.6684837
2: 1.4881108, 24.2587891, 1.5062466, 24.2983627, -20.4411049, 20.3981552
3: -12.8879824, 14.5966396, -12.9057369, 14.6499443, -23.5624619, 23.5738220
4: -1.8226688, 26.1700401, -1.7786639, 26.2131500, -24.3232727, 24.2198372
5: 0.2964735, 23.3055916, 0.2836599, 23.3379173, -20.6639023, 20.6869431
6: -74.4573669, -34.1032448, -74.5451431, -34.1358604, -29.0264435, 29.2057037
7: -4.9927006, 19.2316132, -4.9849010, 19.2762070, -20.0450134, 20.0012703
8: -8.8486538, 20.4906006, -8.8267841, 20.5438995, -27.9491310, 27.8920021
9: -26.2112389, 10.8551626, -26.2517719, 10.8918190, -30.5205498, 30.5384827
10: -35.3416901, 10.8729620, -35.4087143, 10.8702230, -37.0270233, 37.0232544
11: -24.9715405, 9.8987570, -24.9861946, 9.8676739, -27.4293442, 27.4502449
12: -42.4555054, -6.0135150, -42.5267677, -6.0424852, -25.9027634, 25.9796143
13: -38.4403534, 20.1360130, -38.4897308, 20.1708775, -43.2011871, 43.2158546
14: 1.9463224, 49.0037727, 1.9026489, 49.0905952, -39.4253387, 39.3912621
15: -16.9326973, 19.4115982, -16.9122543, 19.4230385, -31.8255768, 31.7596283
16: -38.0402679, 6.6421394, -38.1157074, 6.6882429, -34.1612129, 34.2368622
17: 10.0742664, 55.0263062, 9.9836159, 55.0935211, -35.8436737, 35.7847939
18: -8.1277342, 38.7010460, -8.1141405, 38.7406311, -41.4064178, 41.2097321
19: -18.1044788, 17.3443375, -18.1623840, 17.3727779, -30.6135101, 30.5567055
20: -16.1043320, 18.6613884, -16.1507854, 18.7009125, -31.7935104, 31.8129578
21: -16.0288010, 19.2425098, -16.0634460, 19.2550774, -33.3302612, 33.3407516
22: -2.3533583, 31.4531460, -2.3829184, 31.4648857, -27.5373535, 27.5510368
23: -14.8775921, 15.2504110, -14.9345512, 15.2806644, -28.5670319, 28.5246277
24: -8.7725592, 29.2123661, -8.8073645, 29.2393856, -33.4638596, 33.3860245
25: 1.2392540, 38.9613647, 1.2093859, 38.9810028, -34.1656189, 34.1184082
26: -16.2159538, 24.2795906, -16.2370853, 24.3263798, -36.6003265, 36.5610085
27: -14.3141613, 21.6311989, -14.3054543, 21.6800842, -26.5169983, 26.6006317
28: -5.4661884, 23.8584042, -5.4888096, 23.8919811, -29.1134644, 29.0915222
29: -6.2480183, 24.1766548, -6.2926960, 24.1724796, -26.0377541, 26.0935402
30: -12.4704199, 22.0425415, -12.5068445, 22.0475082, -30.2848816, 30.3689880
31: -17.1947803, 27.1942005, -17.2619762, 27.2168427, -39.3280563, 39.2208328
32: -70.1025085, -28.9819622, -70.1840897, -29.0033398, -30.5916824, 30.7773361
33: -110.2064514, -44.8538284, -110.3606873, -44.8508224, -45.1608734, 45.3242569
34: -88.6744385, -42.1917496, -88.7981262, -42.2069016, -28.9034348, 29.0735054
35: -66.4819260, -20.9674549, -66.5789948, -20.9755249, -30.7008286, 30.8149529
36: -60.7547302, -16.1419945, -60.8272095, -16.1167679, -29.4236679, 29.4303436
37: -121.2674789, -55.2989159, -121.4551315, -55.2926140, -45.9130554, 46.0844269
38: -70.6033401, -15.7077847, -70.6860657, -15.6822472, -39.6498718, 39.6622162
39: -110.9500504, -44.5192871, -111.0790863, -44.5159836, -41.2985535, 41.3921585
40: -114.7776184, -64.8701096, -114.8949127, -64.8840942, -35.5753021, 35.5639610
41: -87.1166229, -42.6246567, -87.2453918, -42.6126175, -29.2852249, 29.4944458
42: -72.5114288, -36.0326080, -72.5954208, -36.0513268, -29.9656448, 30.0721283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=324, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.0609577, upper bound: 24.2455833
time: 41.87 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2049630, upper bound: 24.3205399
time: 37.21 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -22.1920643, 15.9203224, -22.1756744, 15.9265289, -33.8082123, 33.6873779
1: -2.0289886, 26.0799217, -1.9185822, 26.1049156, -23.8742981, 23.7219620
2: 1.4244752, 24.3117599, 1.5053132, 24.3143692, -20.5224876, 20.4519424
3: -12.9223614, 14.6430683, -12.9071941, 14.6622982, -23.6126976, 23.6228142
4: -1.9044588, 26.2328701, -1.7801964, 26.2319527, -24.4284630, 24.2821655
5: 0.2460976, 23.3425541, 0.2825456, 23.3476486, -20.7292938, 20.7244911
6: -74.5409698, -34.0275154, -74.5658264, -34.1356850, -29.1034508, 29.2883568
7: -5.0523310, 19.2723351, -4.9864149, 19.2875500, -20.1179237, 20.0439472
8: -8.9868441, 20.5471134, -8.8280153, 20.5631313, -28.1167717, 27.9474640
9: -26.2839870, 10.9282894, -26.2525711, 10.9114437, -30.6243439, 30.6117172
10: -35.3805046, 10.9319811, -35.4108429, 10.8802290, -37.1130066, 37.0863647
11: -25.0044022, 9.9334555, -24.9893208, 9.8685207, -27.4683495, 27.5241432
12: -42.5449409, -5.9288568, -42.5565414, -6.0416093, -25.9811401, 26.1001053
13: -38.5234528, 20.3005085, -38.5162659, 20.1750088, -43.2760162, 43.4158478
14: 1.8202791, 49.0487289, 1.8940353, 49.1029701, -39.5598564, 39.4426270
15: -17.0332203, 19.4829597, -16.9134254, 19.4406281, -31.9645538, 31.8266220
16: -38.1777267, 6.7101984, -38.1171951, 6.7070718, -34.3322449, 34.3003082
17: 9.9232759, 55.1215935, 9.9435396, 55.0938721, -35.9868240, 35.9262619
18: -8.2290611, 38.7715340, -8.1184101, 38.7643776, -41.5360489, 41.2805176
19: -18.1438446, 17.4135170, -18.1688423, 17.3735676, -30.6488037, 30.6680984
20: -16.1535835, 18.7390728, -16.1638069, 18.7031174, -31.8425369, 31.9215469
21: -16.0815964, 19.3118553, -16.0707493, 19.2571487, -33.3701706, 33.4723816
22: -2.4149213, 31.5473480, -2.3996601, 31.4665966, -27.5995216, 27.6683617
23: -14.9108486, 15.2955446, -14.9397926, 15.2820683, -28.5906982, 28.6353760
24: -8.8057003, 29.2505665, -8.8099222, 29.2411575, -33.4941635, 33.4746857
25: 1.1870117, 39.0348854, 1.2037697, 38.9837189, -34.2039719, 34.2758675
26: -16.2811871, 24.3060665, -16.2441463, 24.3286171, -36.6490250, 36.6441803
27: -14.4232626, 21.6815090, -14.3090754, 21.6957493, -26.6507797, 26.6493340
28: -5.5032892, 23.9144497, -5.4951982, 23.8943634, -29.1469345, 29.2047462
29: -6.3179531, 24.2513695, -6.3111134, 24.1739311, -26.1081543, 26.1897888
30: -12.5549316, 22.1513329, -12.5257273, 22.0497437, -30.3656654, 30.4990768
31: -17.2466221, 27.2331257, -17.2681580, 27.2179642, -39.3645401, 39.3340454
32: -70.1731949, -28.8764877, -70.2046585, -29.0014229, -30.6624908, 30.9080334
33: -110.3158264, -44.6666985, -110.3967743, -44.8488312, -45.2484741, 45.5724258
34: -88.7447052, -42.0579109, -88.8190765, -42.2061615, -28.9649315, 29.2303238
35: -66.5547638, -20.8470497, -66.6023407, -20.9750061, -30.7628174, 30.9723587
36: -60.8195038, -16.0363655, -60.8481445, -16.1160107, -29.4824066, 29.5623360
37: -121.3798141, -55.2099953, -121.4890442, -55.2918739, -46.0151825, 46.2315369
38: -70.6754150, -15.6426725, -70.7047424, -15.6816921, -39.7190399, 39.7614670
39: -111.0457535, -44.3763199, -111.1098785, -44.5152206, -41.3801422, 41.5828476
40: -114.8331604, -64.8406982, -114.9025269, -64.8839493, -35.6675262, 35.5864067
41: -87.1775360, -42.5526581, -87.2572403, -42.6114311, -29.3555145, 29.5724335
42: -72.5757523, -35.9723167, -72.6066818, -36.0501671, -30.0264130, 30.1451225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=324, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1430
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1735861, upper bound: 24.1762578
time: 26.83 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2476870, upper bound: 24.3205708
time: 45.11 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.1623325, 15.8636570, -22.1717339, 15.9289379, -33.7539062, 33.6370621
1: -1.9331801, 26.0277615, -1.9492364, 26.0956688, -23.7657890, 23.6983910
2: 1.4741383, 24.2591362, 1.4721475, 24.3149796, -20.4647713, 20.4270458
3: -12.8875246, 14.6065149, -12.9140568, 14.6539040, -23.5846062, 23.5784836
4: -1.8547170, 26.1715641, -1.8514316, 26.2346306, -24.3759155, 24.2806015
5: 0.2978029, 23.3097782, 0.2745194, 23.3422947, -20.7336960, 20.6502876
6: -74.4570084, -34.0812950, -74.5658646, -34.0762138, -29.1226997, 29.1978760
7: -4.9862041, 19.2441730, -5.0000191, 19.2781429, -20.0341644, 20.0358906
8: -8.8632507, 20.4923668, -8.8673029, 20.5569305, -27.9457321, 27.9415817
9: -26.2242565, 10.8576765, -26.2456684, 10.9167328, -30.6008415, 30.4945221
10: -35.3712311, 10.8867855, -35.3952103, 10.9175224, -37.1396255, 36.9744759
11: -24.9638538, 9.9248495, -25.0139275, 9.9402580, -27.4673843, 27.5188828
12: -42.4590607, -5.9666185, -42.5569687, -5.9876351, -25.9860649, 26.0086632
13: -38.4384689, 20.2259617, -38.5145760, 20.1939926, -43.1769257, 43.3528061
14: 1.9355698, 49.0676193, 1.8620615, 49.0857239, -39.4253922, 39.4860420
15: -16.9813232, 19.4039268, -16.9750614, 19.4544811, -31.9154434, 31.7902756
16: -38.0931816, 6.6497822, -38.0708847, 6.7078323, -34.2883301, 34.1557350
17: 10.0694008, 55.0971336, 9.9396009, 55.1012497, -35.8058167, 35.9176216
18: -8.1861525, 38.6996346, -8.1775265, 38.7668686, -41.4755554, 41.2577896
19: -18.1261063, 17.3427162, -18.1543427, 17.3804703, -30.5823822, 30.6277924
20: -16.1041470, 18.7035656, -16.1705208, 18.7158813, -31.7979813, 31.8805428
21: -16.0304680, 19.2562218, -16.0803833, 19.2851505, -33.3355026, 33.4082642
22: -2.3601294, 31.4507122, -2.4038553, 31.4651737, -27.5413818, 27.5797462
23: -14.8984528, 15.2508144, -14.9273386, 15.2891102, -28.4806900, 28.6584015
24: -8.7943201, 29.2120018, -8.8349476, 29.2500992, -33.4580536, 33.4479828
25: 1.2374535, 38.9691887, 1.1913919, 38.9945526, -34.1172485, 34.2113838
26: -16.2165184, 24.2802124, -16.2619209, 24.3249016, -36.5882568, 36.6194000
27: -14.3390894, 21.6287994, -14.3572235, 21.6958332, -26.5575485, 26.6397533
28: -5.4732485, 23.8590660, -5.4945641, 23.8974648, -29.0792694, 29.1744232
29: -6.2484531, 24.1894665, -6.3129444, 24.1986618, -26.0533752, 26.1242924
30: -12.4696636, 22.1062813, -12.5329704, 22.0889950, -30.3038521, 30.4606056
31: -17.2245865, 27.1921062, -17.2614040, 27.2232533, -39.2920074, 39.2862320
32: -70.1043396, -28.9670353, -70.1999207, -28.9603653, -30.6568527, 30.7771950
33: -110.2665558, -44.8559265, -110.3693695, -44.8188477, -45.2394714, 45.3142548
34: -88.7289963, -42.1952782, -88.7980804, -42.1795616, -29.0183716, 29.0283089
35: -66.5158386, -20.9719868, -66.5899582, -20.9499283, -30.7568741, 30.8085175
36: -60.7585754, -16.1372986, -60.8390198, -16.1129589, -29.4421654, 29.4348450
37: -121.3683167, -55.3040581, -121.4380493, -55.2575607, -46.0708313, 45.9923401
38: -70.6132660, -15.7075729, -70.7105179, -15.6710215, -39.6877289, 39.6660690
39: -110.9893646, -44.5255737, -111.0931244, -44.4819107, -41.3019714, 41.4302826
40: -114.8637543, -64.8737869, -114.8770370, -64.8515015, -35.7617416, 35.4480476
41: -87.1803131, -42.6227722, -87.2260437, -42.5855980, -29.4187698, 29.4036961
42: -72.5285873, -36.0124626, -72.5938721, -35.9967499, -30.0028000, 30.1354446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=324, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1659761, upper bound: 24.1758645
time: 51.99 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2409398, upper bound: 24.3205399
time: 42.39 seconds

## BFS IS instance: IS_A1_B2_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.2110519, 15.9211988, -22.2220116, 15.9469881, -33.8227997, 33.7443924
1: -2.0435047, 26.0783997, -1.9493859, 26.1197166, -23.9008446, 23.7454071
2: 1.4092870, 24.3116093, 1.4724526, 24.3315163, -20.5533829, 20.4736614
3: -12.9240742, 14.6391096, -12.9133244, 14.6800861, -23.6538963, 23.6076050
4: -1.9355347, 26.2338562, -1.8539653, 26.2539673, -24.4806976, 24.3433533
5: 0.2456651, 23.3415260, 0.2751923, 23.3572140, -20.7926369, 20.6915131
6: -74.5392685, -34.0033188, -74.5879669, -34.0782738, -29.1861191, 29.2925682
7: -5.0512109, 19.2724724, -4.9961414, 19.3019218, -20.1400757, 20.0420456
8: -9.0025940, 20.5472908, -8.8674288, 20.5778580, -28.1430511, 27.9673080
9: -26.2817154, 10.9325399, -26.2617741, 10.9346085, -30.6671448, 30.6046638
10: -35.3757286, 10.9457159, -35.4316635, 10.9276676, -37.1599274, 37.1010208
11: -25.0039101, 9.9651814, -25.0095253, 9.9355087, -27.5090904, 27.5900650
12: -42.5460625, -5.8990402, -42.5891953, -5.9696083, -26.0879974, 26.1055527
13: -38.5230331, 20.3342972, -38.5397263, 20.2543049, -43.3187714, 43.4812164
14: 1.8145285, 49.0664902, 1.8484192, 49.1441956, -39.6140251, 39.4812050
15: -17.0696335, 19.4815578, -16.9884720, 19.4653473, -32.0108643, 31.8992500
16: -38.1719589, 6.7191734, -38.1310272, 6.7252483, -34.3783646, 34.2972069
17: 9.9200191, 55.1465111, 9.8987827, 55.1475220, -36.0220413, 35.9857864
18: -8.2639380, 38.7714462, -8.2054482, 38.7892876, -41.5853577, 41.3484497
19: -18.1373882, 17.4148483, -18.1888924, 17.3781528, -30.5766449, 30.7729225
20: -16.1540432, 18.7572250, -16.1829033, 18.7421570, -31.8808136, 31.9542923
21: -16.0788555, 19.3260651, -16.0920982, 19.2867317, -33.3841400, 33.5305634
22: -2.4215679, 31.5465775, -2.4207010, 31.4652214, -27.6016045, 27.6992722
23: -14.9121628, 15.2962637, -14.9521379, 15.2901449, -28.5361557, 28.7359238
24: -8.8176222, 29.2510185, -8.8474693, 29.2510490, -33.4766693, 33.5480766
25: 1.1853275, 39.0417023, 1.1856060, 38.9982986, -34.1609802, 34.3635597
26: -16.2814560, 24.3054924, -16.2693119, 24.3283768, -36.6385498, 36.7008209
27: -14.4437485, 21.6811523, -14.3653364, 21.7094822, -26.6881866, 26.6915970
28: -5.5039196, 23.9153442, -5.5073853, 23.8995514, -29.1308632, 29.2679024
29: -6.3173714, 24.2636814, -6.3324432, 24.2006454, -26.1165886, 26.2277946
30: -12.5546198, 22.1853638, -12.5507936, 22.1208496, -30.4267349, 30.5459404
31: -17.2479458, 27.2332039, -17.2960949, 27.2222290, -39.2926331, 39.4352875
32: -70.1730118, -28.8629932, -70.2225342, -28.9569950, -30.7286987, 30.9067993
33: -110.3365326, -44.6660309, -110.4448395, -44.8196411, -45.2571640, 45.6182785
34: -88.7596512, -42.0598793, -88.8587646, -42.1802902, -29.0240211, 29.2374954
35: -66.5672455, -20.8474083, -66.6347427, -20.9535828, -30.7734528, 31.0052109
36: -60.8212357, -16.0367126, -60.8620834, -16.1071835, -29.5005341, 29.5671883
37: -121.4017944, -55.2099304, -121.5506897, -55.2621307, -46.0284576, 46.2733994
38: -70.6821289, -15.6438541, -70.7322845, -15.6689978, -39.7395935, 39.7811356
39: -111.0600357, -44.3761406, -111.1489716, -44.4875755, -41.3436279, 41.6565475
40: -114.8538818, -64.8391418, -114.9500427, -64.8565521, -35.7475891, 35.5712662
41: -87.1905975, -42.5498466, -87.2885132, -42.5853577, -29.4101639, 29.5574493
42: -72.5767059, -35.9525528, -72.6213455, -35.9952278, -30.0713196, 30.1995087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=325, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=206, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1431
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1003
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1454
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1421
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1535
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1503
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 880
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 993
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1374
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1501
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1291
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1514
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1429
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1323
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1286
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1290
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2090561, upper bound: 24.1762578
time: 37.75 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.2836877, upper bound: 24.3205708
time: 95.92 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -22.2398224, 15.9749870, -22.2571850, 15.9208870, -33.7639656, 33.8365898
1: -1.9538791, 26.1160355, -2.0456622, 26.0795670, -23.7581444, 23.8956718
2: 1.4744835, 24.3206673, 1.4230814, 24.3150024, -20.4797020, 20.5587425
3: -12.9196997, 14.7024412, -12.9428749, 14.6476250, -23.5899048, 23.7002048
4: -1.8518691, 26.2424679, -1.9183807, 26.2439461, -24.3543510, 24.4528275
5: 0.2662730, 23.3789845, 0.2315283, 23.3440933, -20.7098579, 20.8258171
6: -74.5587769, -34.0819283, -74.5414047, -34.0269966, -29.2355423, 29.1984825
7: -4.9918346, 19.2915039, -5.0375671, 19.2759972, -20.0428619, 20.1287346
8: -8.9074612, 20.5872269, -8.9984684, 20.5765476, -28.0369072, 28.1873779
9: -26.2758961, 10.9891539, -26.3148613, 10.9404697, -30.6395569, 30.7526283
10: -35.4398346, 11.0017977, -35.4248199, 10.9412174, -37.1982040, 37.2676239
11: -25.0268478, 9.9459782, -25.0095501, 9.9842434, -27.6425591, 27.5263214
12: -42.6002464, -5.9031401, -42.6005173, -5.9044275, -26.1785431, 26.2129517
13: -38.5622292, 20.3157196, -38.5656357, 20.3373833, -43.4786835, 43.4124832
14: 1.8286724, 49.1932907, 1.7696724, 49.0676346, -39.5231247, 39.7429047
15: -17.0070572, 19.4913902, -17.0925293, 19.4861526, -31.9149246, 32.0450439
16: -38.1447601, 6.7598047, -38.2008171, 6.7221189, -34.3093491, 34.4411240
17: 9.8806581, 55.1750717, 9.8765450, 55.1479492, -36.0989609, 36.1431541
18: -8.2151814, 38.7673836, -8.2656364, 38.7717972, -41.3515625, 41.4848022
19: -18.2538052, 17.3957462, -18.1402206, 17.4546299, -30.8862686, 30.5647736
20: -16.2213383, 18.7607727, -16.1543083, 18.7913094, -32.0281143, 31.9066467
21: -16.1346970, 19.3056011, -16.0813332, 19.3595161, -33.6074905, 33.4044800
22: -2.4508324, 31.4742203, -2.4218621, 31.5597420, -27.7317390, 27.5995522
23: -15.0127258, 15.3138885, -14.9156532, 15.3358011, -28.8357086, 28.5948982
24: -8.8879223, 29.2718544, -8.8249207, 29.3017864, -33.6359940, 33.4431686
25: 1.1521320, 39.0148888, 1.1807899, 39.0734482, -34.4296570, 34.1670418
26: -16.3275776, 24.3485031, -16.2839470, 24.3532486, -36.8059082, 36.6564903
27: -14.4591513, 21.7330456, -14.4421043, 21.7370300, -26.8331299, 26.8375492
28: -5.5770235, 23.9201622, -5.5046701, 23.9539413, -29.3717003, 29.1778259
29: -6.3356695, 24.2016907, -6.3194590, 24.2667351, -26.2312241, 26.1193981
30: -12.5643053, 22.1388855, -12.5568247, 22.2050877, -30.5460739, 30.4482574
31: -17.3284473, 27.2385139, -17.2565041, 27.2663784, -39.5069199, 39.2248840
32: -70.1852570, -28.9621220, -70.1760025, -28.8889236, -30.8115997, 30.7469711
33: -110.4399948, -44.8210754, -110.3403015, -44.6774826, -45.5776520, 45.2615204
34: -88.7972107, -42.1952171, -88.7599182, -42.0981255, -29.1629639, 29.0353928
35: -66.6054382, -20.9587250, -66.5727310, -20.8597927, -30.9653625, 30.7844925
36: -60.8823547, -16.1046524, -60.8235703, -16.0308266, -29.6760254, 29.5083237
37: -121.5714264, -55.2586479, -121.4159698, -55.2048950, -46.3175430, 45.9943085
38: -70.7542419, -15.6647425, -70.6879730, -15.6293669, -39.8494415, 39.7038574
39: -111.1714630, -44.4807816, -111.0619659, -44.3636818, -41.6793442, 41.3636932
40: -114.9328156, -64.8639526, -114.8597565, -64.8627167, -35.6282578, 35.6806488
41: -87.2948761, -42.5862198, -87.1929092, -42.5452805, -29.6194534, 29.4765301
42: -72.6049118, -35.9961853, -72.5791397, -35.9688568, -30.1577606, 30.0950851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1322
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1322
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1286
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1757058, upper bound: 24.1787459
time: 58.14 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3200315, upper bound: 24.2532567
time: 36.74 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -22.2478638, 15.9762411, -22.2904758, 15.9581852, -33.8119965, 33.8656540
1: -1.9647121, 26.1165314, -2.0719512, 26.1188679, -23.8089523, 23.9124565
2: 1.4668589, 24.3219852, 1.4042759, 24.3411617, -20.5057182, 20.5721760
3: -12.9262276, 14.7036104, -12.9580803, 14.6749630, -23.6215668, 23.7112865
4: -1.8610048, 26.2437668, -1.9401062, 26.2698059, -24.3968201, 24.4699059
5: 0.2606602, 23.3798561, 0.2168684, 23.3619251, -20.7365074, 20.8396759
6: -74.5608978, -34.0643883, -74.6095276, -33.9872093, -29.2555008, 29.2897873
7: -4.9988489, 19.2922745, -5.0541039, 19.2980423, -20.0606575, 20.1380577
8: -8.9099827, 20.5878944, -9.0077763, 20.5914707, -28.0443497, 28.1980972
9: -26.2828751, 10.9906416, -26.3322983, 10.9592018, -30.6693497, 30.7659569
10: -35.4406357, 11.0083961, -35.4429512, 10.9603558, -37.2255096, 37.2609634
11: -25.0282021, 9.9517393, -25.0340118, 9.9980316, -27.6535416, 27.5535583
12: -42.6017151, -5.9001098, -42.6172371, -5.8952579, -26.1866341, 26.2359276
13: -38.5640526, 20.3187084, -38.5724602, 20.3640575, -43.5188370, 43.4276199
14: 1.8111620, 49.1941261, 1.7196674, 49.1164932, -39.5714340, 39.7804260
15: -17.0076275, 19.4926262, -17.0962257, 19.4950447, -31.9390640, 32.0456772
16: -38.1498413, 6.7606091, -38.2182083, 6.7423334, -34.3377800, 34.4607010
17: 9.8576546, 55.1755409, 9.8206806, 55.2047882, -36.1429138, 36.1743584
18: -8.2211208, 38.7680397, -8.2867575, 38.7967453, -41.3858795, 41.5007782
19: -18.2550392, 17.4035168, -18.1726780, 17.4720707, -30.8975983, 30.5991096
20: -16.2221451, 18.7719269, -16.1967468, 18.8200531, -32.0505219, 31.9600449
21: -16.1364365, 19.3138371, -16.1155853, 19.3779926, -33.6236115, 33.4468651
22: -2.4525704, 31.4771805, -2.4396572, 31.5670929, -27.7425194, 27.6306000
23: -15.0139141, 15.3217163, -14.9500027, 15.3538990, -28.8523254, 28.6380310
24: -8.8898602, 29.2731934, -8.8628731, 29.3067303, -33.6441345, 33.4861412
25: 1.1504984, 39.0192871, 1.1549206, 39.0842209, -34.4415512, 34.2001915
26: -16.3291950, 24.3518257, -16.3138504, 24.3658791, -36.8187485, 36.6896935
27: -14.4616890, 21.7362881, -14.4651070, 21.7449131, -26.8399773, 26.8732872
28: -5.5786428, 23.9259796, -5.5178099, 23.9700050, -29.3886528, 29.2020378
29: -6.3382406, 24.2075195, -6.3510189, 24.2809372, -26.2438774, 26.1580887
30: -12.5661345, 22.1449070, -12.5843840, 22.2201271, -30.5592308, 30.4815407
31: -17.3303986, 27.2436256, -17.3007355, 27.2784405, -39.5155487, 39.2696381
32: -70.1872025, -28.9476662, -70.2314911, -28.8541260, -30.8316879, 30.8244762
33: -110.4409485, -44.7954102, -110.4200211, -44.6183205, -45.6164856, 45.3892136
34: -88.7986298, -42.1737022, -88.8343506, -42.0480385, -29.1895447, 29.1094589
35: -66.6065750, -20.9448395, -66.6256027, -20.8257065, -30.9829102, 30.8577805
36: -60.8841820, -16.0892792, -60.8715477, -15.9946270, -29.7016602, 29.5038834
37: -121.5728760, -55.2331543, -121.5011292, -55.1469536, -46.3443527, 46.0952110
38: -70.7557220, -15.6471701, -70.7503967, -15.5881109, -39.8762589, 39.7518539
39: -111.1731796, -44.4583664, -111.1355667, -44.3170433, -41.7027054, 41.4690170
40: -114.9332275, -64.8429794, -114.9227448, -64.8143311, -35.6529083, 35.6488533
41: -87.2966766, -42.5664978, -87.2662659, -42.4959373, -29.6488113, 29.5393620
42: -72.6063004, -35.9767113, -72.6393433, -35.9234581, -30.1872864, 30.1904716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=325, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=207, inp2_unstable=207, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=9, inp2_unstable=9, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1431
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1339
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 101
type: A, layer: 1, pos: 101
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 73
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1017
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 920
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 957
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 957
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1003
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 951
type: A, layer: 1, pos: 951
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1499
type: B, layer: 1, pos: 1499
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1470
type: B, layer: 1, pos: 1470
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1453
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1439
type: B, layer: 1, pos: 1439
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1454
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1437
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1421
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1454
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1535
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1002
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1503
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1471
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 880
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 993
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1485
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1374
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1437
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1528
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1528
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 919
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 897
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 919
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1530
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 811
type: B, layer: 1, pos: 811
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1530
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1534
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1534
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1515
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 1292
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1292
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1501
type: A, layer: 1, pos: 1519
type: B, layer: 1, pos: 1519
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1293
type: A, layer: 1, pos: 1293
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1484
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1291
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 1486
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1514
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1429
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1323
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1295
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1001
type: A, layer: 1, pos: 1359
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1294
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1359
type: A, layer: 1, pos: 1001
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1289
type: B, layer: 1, pos: 1289
type: A, layer: 1, pos: 1288
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1288
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.1762578, upper bound: 24.2458639
time: 49.85 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 17, lower bound: -24.3205708, upper bound: 24.3205711
time: 44.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 97.21 seconds
IS_A1_B1_B2_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.0609577, upper bound: 24.2455833
IS_A1_B1_B2_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.2049630, upper bound: 24.3205399
IS_A1_B1_B2_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.1735861, upper bound: 24.1762578
IS_A1_B1_B2_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.2476870, upper bound: 24.3205708
IS_A1_B2_A2_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.1659761, upper bound: 24.1758645
IS_A1_B2_A2_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.2409398, upper bound: 24.3205399
IS_A1_B2_A2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.2090561, upper bound: 24.1762578
IS_A1_B2_A2_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.2836877, upper bound: 24.3205708
IS_A2_A2_B2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.1757058, upper bound: 24.1787459
IS_A2_A2_B2_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.3200315, upper bound: 24.2532567
IS_A2_A2_B2_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.1762578, upper bound: 24.2458639
IS_A2_A2_B2_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 97.21
Output dim: 17, lower bound: -24.3205708, upper bound: 24.3205711

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 58.48 + 3541.72 = 3600.21 seconds
