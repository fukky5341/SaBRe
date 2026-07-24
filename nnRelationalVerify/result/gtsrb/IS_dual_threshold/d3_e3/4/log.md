## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 33.4998133448


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=223, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.3687057, 28.3436661, -29.3687057, 28.3436661, -57.7123718, 57.7123718)
1: (-14.3686533, 23.1169357, -14.3686533, 23.1169357, -37.4855881, 37.4855881)
2: (-20.8331318, 23.2205544, -20.8331318, 23.2205544, -43.2646713, 43.2646713)
3: (-27.1577568, 23.1048012, -27.1577568, 23.1048012, -48.4871216, 48.4871140)
4: (-32.6859093, 17.9857941, -32.6859093, 17.9857941, -50.2830048, 50.2830048)
5: (-28.1991749, 25.0023537, -28.1991749, 25.0023537, -53.1612701, 53.1612625)
6: (-41.8412781, 7.6574535, -41.8412781, 7.6574535, -47.5941086, 47.5941124)
7: (-28.7920551, 29.0157146, -28.7920551, 29.0157146, -57.8077698, 57.8077698)
8: (-36.2347603, 23.7399025, -36.2347603, 23.7399025, -58.9520111, 58.9520187)
9: (-22.6215839, 22.7077217, -22.6215839, 22.7077217, -45.3293076, 45.3293076)
10: (-30.6834221, 30.5571270, -30.6834221, 30.5571270, -61.2405472, 61.2405472)
11: (-24.4137306, 25.3683167, -24.4137306, 25.3683167, -49.1342850, 49.1342812)
12: (-46.1738358, 35.1112747, -46.1738358, 35.1112747, -81.2851105, 81.2851105)
13: (-48.3583984, 19.7234249, -48.3583984, 19.7234249, -68.0818253, 68.0818253)
14: (-64.3322525, 26.6960144, -64.3322525, 26.6960144, -89.7105408, 89.7105408)
15: (-34.0794830, 12.3438988, -34.0794830, 12.3438988, -46.4233818, 46.4233818)
16: (-29.4932594, 21.6498260, -29.4932594, 21.6498260, -51.1430855, 51.1430855)
17: (-59.0885239, 28.6373005, -59.0885239, 28.6373005, -87.7258224, 87.7258224)
18: (-28.2209415, 25.2926254, -28.2209415, 25.2926254, -53.5135651, 53.5135651)
19: (-24.2912693, 11.1597157, -24.2912693, 11.1597157, -35.0300903, 35.0300903)
20: (-16.6355476, 16.1454773, -16.6355476, 16.1454773, -32.7810249, 32.7810249)
21: (-23.1503315, 15.7911301, -23.1503315, 15.7911301, -38.9414597, 38.9414597)
22: (-29.8906784, 18.1935844, -29.8906784, 18.1935844, -48.0842628, 48.0842628)
23: (-17.7845211, 20.5397320, -17.7845211, 20.5397320, -37.9173088, 37.9173126)
24: (-22.4685593, 16.5004807, -22.4685593, 16.5004807, -38.0747757, 38.0747757)
25: (-21.8450851, 20.7178764, -21.8450851, 20.7178764, -41.8102646, 41.8102684)
26: (-48.7503052, 23.4067230, -48.7503052, 23.4067230, -72.1570282, 72.1570282)
27: (-27.3533859, 21.0031128, -27.3533859, 21.0031128, -48.2104263, 48.2104263)
28: (-18.8090534, 22.7977848, -18.8090534, 22.7977848, -40.2101440, 40.2101479)
29: (-27.1733360, 20.9911938, -27.1733360, 20.9911938, -47.1226959, 47.1226959)
30: (-21.8117905, 18.9658546, -21.8117905, 18.9658546, -39.9472885, 39.9472923)
31: (-25.4966908, 18.5137157, -25.4966908, 18.5137157, -44.0104065, 44.0104065)
32: (-33.1184769, 16.2344227, -33.1184769, 16.2344227, -49.2135391, 49.2135315)
33: (-63.7556000, 12.8170786, -63.7556000, 12.8170786, -74.6543121, 74.6543198)
34: (-47.6960907, 8.2922297, -47.6960907, 8.2922297, -51.6974487, 51.6974564)
35: (-45.0071297, 11.5211735, -45.0071297, 11.5211735, -54.6439056, 54.6439056)
36: (-51.4329529, 13.5137806, -51.4329529, 13.5137806, -64.9467316, 64.9467316)
37: (-67.4206390, 1.4809723, -67.4206390, 1.4809723, -66.6819916, 66.6819992)
38: (-54.0977440, 16.3466244, -54.0977440, 16.3466244, -70.4443665, 70.4443665)
39: (-73.4575043, 1.7182684, -73.4575043, 1.7182684, -75.0631104, 75.0631104)
40: (-58.5168610, 0.1258335, -58.5168610, 0.1258335, -55.5951614, 55.5951691)
41: (-42.3647079, 12.1001272, -42.3647079, 12.1001272, -50.3985748, 50.3985901)
42: (-26.1969643, 15.6938057, -26.1969643, 15.6938057, -38.9739914, 38.9739876)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.99 + 47.57 = 50.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -33.5199253, upper bound: 33.5199253

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 665
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1415
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 1518
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 952
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 1447
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 1378
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.5027405, upper bound: 33.4668397
time: 34.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.5027405, upper bound: 33.5027404
time: 29.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 63.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 63.29
Output dim: 11, lower bound: -33.5027405, upper bound: 33.4668397
IS_A2, status: Status.UNKNOWN, split count: 1, time: 63.29
Output dim: 11, lower bound: -33.5027405, upper bound: 33.5027404

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.1597099, 28.2779770, -29.2646847, 28.3105793, -57.4702911, 57.5426636
1: -14.2893314, 23.0857410, -14.3291283, 23.1012383, -37.3905716, 37.4148712
2: -20.6399593, 23.1779404, -20.7370453, 23.1987038, -43.0496140, 43.1252518
3: -26.9894981, 23.0473576, -27.0742779, 23.0759010, -48.2875519, 48.3437347
4: -32.4450073, 17.9417686, -32.5660477, 17.9638596, -50.0151672, 50.1155243
5: -27.9946308, 24.9245834, -28.0977135, 24.9628525, -52.9146881, 52.9791489
6: -41.7787247, 7.5050697, -41.8102150, 7.5796947, -47.4195023, 47.3597908
7: -28.6098785, 28.9631042, -28.7014713, 28.9887199, -57.5985985, 57.6645737
8: -36.0051765, 23.6841850, -36.1207199, 23.7116871, -58.6938477, 58.7832870
9: -22.5393467, 22.6074371, -22.5799446, 22.6577911, -45.1971359, 45.1873817
10: -30.6074753, 30.4390793, -30.6453247, 30.4982929, -61.1057663, 61.0844040
11: -24.3154392, 25.1822014, -24.3648796, 25.2732639, -48.9447632, 48.9026566
12: -46.0940094, 34.5548363, -46.1332932, 34.8350754, -80.9290848, 80.6881256
13: -48.2901077, 19.4602547, -48.3240242, 19.5896854, -67.8797913, 67.7842789
14: -64.2169724, 26.2474213, -64.2745132, 26.4679813, -89.3574219, 89.1814804
15: -33.9229050, 12.2647095, -34.0008965, 12.3046837, -46.2275887, 46.2656059
16: -29.3895493, 21.5942822, -29.4416656, 21.6221275, -51.0116768, 51.0359497
17: -59.0010376, 28.1789627, -59.0449066, 28.4044189, -87.4054565, 87.2238693
18: -28.1243725, 25.1921520, -28.1719666, 25.2420464, -53.3664169, 53.3641205
19: -24.2038288, 11.1424103, -24.2475624, 11.1510668, -34.9307556, 34.9671555
20: -16.5569420, 16.0598602, -16.5963078, 16.1019268, -32.6588669, 32.6561661
21: -23.0631828, 15.7560596, -23.1066990, 15.7730675, -38.8362503, 38.8627586
22: -29.8071766, 18.1107349, -29.8487473, 18.1523132, -47.9594879, 47.9594803
23: -17.7171612, 20.5110226, -17.7505341, 20.5249786, -37.8219872, 37.8461151
24: -22.2905445, 16.4692039, -22.3799992, 16.4847412, -37.8755264, 37.9514771
25: -21.7271156, 20.6676140, -21.7856541, 20.6927948, -41.6617661, 41.6950493
26: -48.6552353, 23.2002563, -48.7022667, 23.3039894, -71.9592285, 71.9025269
27: -27.1996250, 20.9611340, -27.2751122, 20.9820442, -48.0275497, 48.0845871
28: -18.7343025, 22.7657909, -18.7716560, 22.7814465, -40.1137581, 40.1322937
29: -27.1176682, 20.8355808, -27.1454830, 20.9136143, -46.9867477, 46.9356232
30: -21.7373924, 18.8432007, -21.7747498, 18.9037609, -39.8095627, 39.7867355
31: -25.3442669, 18.4822330, -25.4198093, 18.4976845, -43.8419495, 43.9020424
32: -33.0559616, 15.9709806, -33.0872116, 16.1033401, -49.0174789, 48.9094009
33: -63.4937897, 12.7410135, -63.6245193, 12.7790651, -74.3305817, 74.4354706
34: -47.6151276, 8.2278757, -47.6549339, 8.2599430, -51.5665588, 51.5804062
35: -44.8859406, 11.4757137, -44.9454842, 11.4984741, -54.4790039, 54.5242767
36: -51.3720055, 13.3642941, -51.4022293, 13.4382076, -64.8102112, 64.7665253
37: -67.2522125, 1.4265871, -67.3341904, 1.4538231, -66.4634094, 66.5242462
38: -54.0179253, 16.2251244, -54.0578384, 16.2859879, -70.3039093, 70.2829590
39: -73.2773666, 1.6712866, -73.3661346, 1.6946487, -74.8548737, 74.9230652
40: -58.3647614, 0.0772209, -58.4396515, 0.1015167, -55.3923340, 55.4530945
41: -42.2705193, 12.0341997, -42.3176498, 12.0669098, -50.2675858, 50.2842636
42: -26.1321297, 15.5864973, -26.1646652, 15.6397800, -38.8535461, 38.8282814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=222, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 664
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 1467
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1399
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1415
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1385
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1385
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 937
type: B, layer: 1, pos: 1463
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1463
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 937
type: A, layer: 1, pos: 940
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 981
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1346
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1529
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4493716
time: 44.54 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4493716
time: 33.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.4021034, 28.5390606, -29.3522186, 28.3393841, -57.7414856, 57.8912811
1: -14.4066658, 23.2543793, -14.3583345, 23.1138859, -37.5205536, 37.6127129
2: -20.8397026, 23.4236774, -20.8205299, 23.2172241, -43.2606354, 43.4595947
3: -27.1632309, 23.3153877, -27.1468391, 23.0995579, -48.4864197, 48.6941147
4: -32.6856689, 18.2883835, -32.6691399, 17.9822922, -50.2699127, 50.5781403
5: -28.1965046, 25.2713299, -28.1860809, 24.9959583, -53.1434021, 53.4179077
6: -41.9586639, 7.6788540, -41.8360939, 7.6326847, -47.7159348, 47.6073341
7: -28.8269634, 29.1345921, -28.7736130, 29.0117607, -57.8387222, 57.9082031
8: -36.2460365, 23.9693069, -36.2186890, 23.7348328, -58.9487915, 59.1715088
9: -22.7066612, 22.7725315, -22.6127892, 22.6960869, -45.4027481, 45.3853226
10: -30.9535618, 30.6214905, -30.6767216, 30.5387039, -61.4922638, 61.2982101
11: -24.8099785, 25.3480759, -24.4061260, 25.3426590, -49.4987411, 49.1014328
12: -46.7006149, 35.1010437, -46.1685066, 35.0778847, -81.7785034, 81.2695465
13: -48.5176010, 19.7493591, -48.3517036, 19.6955013, -68.2131042, 68.1010590
14: -64.7243500, 26.6884594, -64.3220062, 26.6700840, -90.1032867, 89.6854248
15: -34.0681419, 12.5487928, -34.0425453, 12.3380022, -46.4061432, 46.5913391
16: -29.6932049, 21.7162647, -29.4824905, 21.6446438, -51.3378487, 51.1987534
17: -59.6397476, 28.6301098, -59.0822105, 28.6068172, -88.2465668, 87.7123184
18: -28.3069744, 25.3238335, -28.1989403, 25.2825928, -53.5895691, 53.5227737
19: -24.4332867, 11.2584381, -24.2835541, 11.1581974, -35.1756287, 35.1218185
20: -16.7891655, 16.1652107, -16.6289597, 16.1388321, -32.9279976, 32.7941704
21: -23.4153156, 15.8121071, -23.1420784, 15.7847481, -39.2000656, 38.9541855
22: -29.9974651, 18.2589302, -29.8834057, 18.1818066, -48.1792717, 48.1423340
23: -17.9280739, 20.5964546, -17.7787418, 20.5365524, -38.0439148, 37.9897499
24: -22.5089149, 16.6515427, -22.4566574, 16.4969406, -38.0966110, 38.2174606
25: -21.9127922, 20.7969818, -21.8280849, 20.7131767, -41.8740463, 41.8800430
26: -49.0557480, 23.4108658, -48.7417030, 23.3864002, -72.4421463, 72.1525726
27: -27.3914700, 21.0513725, -27.3330250, 20.9988937, -48.2467346, 48.2629776
28: -18.9345264, 22.8091049, -18.8028831, 22.7938309, -40.3355103, 40.2221870
29: -27.3662548, 21.0002480, -27.1682320, 20.9781418, -47.3024063, 47.1205978
30: -21.9252300, 18.9707813, -21.8050404, 18.9478226, -40.0475807, 39.9430466
31: -25.5996418, 18.6775131, -25.4813519, 18.5106106, -44.1102524, 44.1588669
32: -33.3043671, 16.2387123, -33.1136169, 16.2173233, -49.4060287, 49.2045288
33: -63.7901993, 13.0668106, -63.7343445, 12.8098803, -74.6649017, 74.9049072
34: -47.7379799, 8.4308949, -47.6884384, 8.2840452, -51.7143250, 51.8622894
35: -45.0443916, 11.6257000, -44.9960823, 11.5170317, -54.6536865, 54.7617798
36: -51.6136932, 13.5480938, -51.4279785, 13.5032043, -65.1168976, 64.9760742
37: -67.5130539, 1.5431633, -67.4012299, 1.4766102, -66.7495880, 66.7337494
38: -54.2817001, 16.4099846, -54.0916290, 16.3324165, -70.6141205, 70.5016174
39: -73.5283737, 1.9187231, -73.4383698, 1.7128725, -75.1243134, 75.2426147
40: -58.5486259, 0.2178621, -58.4945221, 0.1201668, -55.6195526, 55.6747894
41: -42.4422646, 12.1237030, -42.3577881, 12.0891342, -50.4649277, 50.4191208
42: -26.3233070, 15.6934099, -26.1926155, 15.6681490, -39.0938911, 38.9708138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=222, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 923
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: A, layer: 1, pos: 760
type: B, layer: 1, pos: 760
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 839
type: A, layer: 1, pos: 839
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 984
type: A, layer: 1, pos: 984
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 665
type: B, layer: 1, pos: 983
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 696
type: A, layer: 1, pos: 696
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 632
type: B, layer: 1, pos: 1450
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1450
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 886
type: B, layer: 1, pos: 886
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1399
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1397
type: A, layer: 1, pos: 1397
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1469
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 890
type: B, layer: 1, pos: 890
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 956
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1482
type: B, layer: 1, pos: 1482
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1303
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 871
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 870
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 870
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 938
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 795
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 795
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 989
type: A, layer: 1, pos: 989
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1447
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 996
type: B, layer: 1, pos: 996
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 982
type: B, layer: 1, pos: 982
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 798
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1518
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 938
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1483
type: B, layer: 1, pos: 1483
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 952
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1403
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 981
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1524
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1346
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1349
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1524
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1529
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 1378
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1446
type: A, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 896
type: B, layer: 1, pos: 896

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4855635
time: 30.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4855635
time: 29.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 62.39 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 62.39
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4493716
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 62.39
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4493716
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 62.39
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4855635
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 62.39
Output dim: 11, lower bound: -33.4294272, upper bound: 33.4855635

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 50.56 + 206.43 = 256.99 seconds
