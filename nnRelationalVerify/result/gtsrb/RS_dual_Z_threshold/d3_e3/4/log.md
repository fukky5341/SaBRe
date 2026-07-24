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
execution time: IAR + RelationalAnalysis = 3.44 + 48.94 = 52.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -33.5199253, upper bound: 33.5199253

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1349

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.5038545, upper bound: 33.4472876
time: 36.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.4472876, upper bound: 33.5038545
time: 43.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 79.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 79.90
Output dim: 11, lower bound: -33.5038545, upper bound: 33.4472876
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 79.90
Output dim: 11, lower bound: -33.4472876, upper bound: 33.5038545

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -29.3687057, 28.3436661, -29.3687057, 28.3436661, -57.7123718, 57.7123718
1: -14.3686533, 23.1169357, -14.3686533, 23.1169357, -37.4855881, 37.4855881
2: -20.8331318, 23.2205544, -20.8331318, 23.2205544, -43.2679901, 43.2635193
3: -27.1577568, 23.1048012, -27.1577568, 23.1048012, -48.4662781, 48.4730148
4: -32.6859093, 17.9857941, -32.6859093, 17.9857941, -50.2880707, 50.2811279
5: -28.1991749, 25.0023537, -28.1991749, 25.0023537, -53.1579132, 53.1587372
6: -41.8412781, 7.6574535, -41.8412781, 7.6574535, -47.5903931, 47.5967560
7: -28.7920551, 29.0157146, -28.7920551, 29.0157146, -57.8077698, 57.8077698
8: -36.2347603, 23.7399025, -36.2347603, 23.7399025, -58.9496918, 58.9501343
9: -22.6215839, 22.7077217, -22.6215839, 22.7077217, -45.3293076, 45.3293076
10: -30.6834221, 30.5571270, -30.6834221, 30.5571270, -61.2405472, 61.2405472
11: -24.4137306, 25.3683167, -24.4137306, 25.3683167, -49.1261292, 49.1228638
12: -46.1738358, 35.1112747, -46.1738358, 35.1112747, -81.2851105, 81.2851105
13: -48.3583984, 19.7234249, -48.3583984, 19.7234249, -68.0818253, 68.0818253
14: -64.3322525, 26.6960144, -64.3322525, 26.6960144, -89.7089386, 89.7090454
15: -34.0794830, 12.3438988, -34.0794830, 12.3438988, -46.4233818, 46.4233818
16: -29.4932594, 21.6498260, -29.4932594, 21.6498260, -51.1430855, 51.1430855
17: -59.0885239, 28.6373005, -59.0885239, 28.6373005, -87.7258224, 87.7258224
18: -28.2209415, 25.2926254, -28.2209415, 25.2926254, -53.5135651, 53.5135651
19: -24.2912693, 11.1597157, -24.2912693, 11.1597157, -35.0337601, 35.0272751
20: -16.6355476, 16.1454773, -16.6355476, 16.1454773, -32.7810249, 32.7810249
21: -23.1503315, 15.7911301, -23.1503315, 15.7911301, -38.9414597, 38.9414597
22: -29.8906784, 18.1935844, -29.8906784, 18.1935844, -48.0842628, 48.0842628
23: -17.7845211, 20.5397320, -17.7845211, 20.5397320, -37.9173355, 37.9161530
24: -22.4685593, 16.5004807, -22.4685593, 16.5004807, -38.0774765, 38.0743332
25: -21.8450851, 20.7178764, -21.8450851, 20.7178764, -41.8011856, 41.8004799
26: -48.7503052, 23.4067230, -48.7503052, 23.4067230, -72.1570282, 72.1570282
27: -27.3533859, 21.0031128, -27.3533859, 21.0031128, -48.1878052, 48.1854935
28: -18.8090534, 22.7977848, -18.8090534, 22.7977848, -40.2060127, 40.2042389
29: -27.1733360, 20.9911938, -27.1733360, 20.9911938, -47.1226044, 47.1224670
30: -21.8117905, 18.9658546, -21.8117905, 18.9658546, -39.9459610, 39.9446564
31: -25.4966908, 18.5137157, -25.4966908, 18.5137157, -44.0104065, 44.0104065
32: -33.1184769, 16.2344227, -33.1184769, 16.2344227, -49.2116470, 49.2141075
33: -63.7556000, 12.8170786, -63.7556000, 12.8170786, -74.6241455, 74.6335907
34: -47.6960907, 8.2922297, -47.6960907, 8.2922297, -51.6810608, 51.6944962
35: -45.0071297, 11.5211735, -45.0071297, 11.5211735, -54.6305542, 54.6324539
36: -51.4329529, 13.5137806, -51.4329529, 13.5137806, -64.9467316, 64.9467316
37: -67.4206390, 1.4809723, -67.4206390, 1.4809723, -66.6788788, 66.6803741
38: -54.0977440, 16.3466244, -54.0977440, 16.3466244, -70.4443665, 70.4443665
39: -73.4575043, 1.7182684, -73.4575043, 1.7182684, -75.0494690, 75.0498505
40: -58.5168610, 0.1258335, -58.5168610, 0.1258335, -55.5833664, 55.5898895
41: -42.3647079, 12.1001272, -42.3647079, 12.1001272, -50.3976288, 50.3980255
42: -26.1969643, 15.6938057, -26.1969643, 15.6938057, -38.9642868, 38.9777527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=223, inp2_unstable=223, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1349

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4953885, upper bound: 33.3993314
time: 30.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4569306, upper bound: 33.4401281
time: 42.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -29.3687057, 28.3436661, -29.3687057, 28.3436661, -57.7123718, 57.7123718
1: -14.3686533, 23.1169357, -14.3686533, 23.1169357, -37.4855881, 37.4855881
2: -20.8331318, 23.2205544, -20.8331318, 23.2205544, -43.2635193, 43.2679939
3: -27.1577568, 23.1048012, -27.1577568, 23.1048012, -48.4730225, 48.4662704
4: -32.6859093, 17.9857941, -32.6859093, 17.9857941, -50.2811279, 50.2880783
5: -28.1991749, 25.0023537, -28.1991749, 25.0023537, -53.1587372, 53.1579056
6: -41.8412781, 7.6574535, -41.8412781, 7.6574535, -47.5967560, 47.5903931
7: -28.7920551, 29.0157146, -28.7920551, 29.0157146, -57.8077698, 57.8077698
8: -36.2347603, 23.7399025, -36.2347603, 23.7399025, -58.9501343, 58.9496994
9: -22.6215839, 22.7077217, -22.6215839, 22.7077217, -45.3293076, 45.3293076
10: -30.6834221, 30.5571270, -30.6834221, 30.5571270, -61.2405472, 61.2405472
11: -24.4137306, 25.3683167, -24.4137306, 25.3683167, -49.1228638, 49.1261253
12: -46.1738358, 35.1112747, -46.1738358, 35.1112747, -81.2851105, 81.2851105
13: -48.3583984, 19.7234249, -48.3583984, 19.7234249, -68.0818253, 68.0818253
14: -64.3322525, 26.6960144, -64.3322525, 26.6960144, -89.7090454, 89.7089386
15: -34.0794830, 12.3438988, -34.0794830, 12.3438988, -46.4233818, 46.4233818
16: -29.4932594, 21.6498260, -29.4932594, 21.6498260, -51.1430855, 51.1430855
17: -59.0885239, 28.6373005, -59.0885239, 28.6373005, -87.7258224, 87.7258224
18: -28.2209415, 25.2926254, -28.2209415, 25.2926254, -53.5135651, 53.5135651
19: -24.2912693, 11.1597157, -24.2912693, 11.1597157, -35.0272751, 35.0337601
20: -16.6355476, 16.1454773, -16.6355476, 16.1454773, -32.7810249, 32.7810249
21: -23.1503315, 15.7911301, -23.1503315, 15.7911301, -38.9414597, 38.9414597
22: -29.8906784, 18.1935844, -29.8906784, 18.1935844, -48.0842628, 48.0842628
23: -17.7845211, 20.5397320, -17.7845211, 20.5397320, -37.9161530, 37.9173355
24: -22.4685593, 16.5004807, -22.4685593, 16.5004807, -38.0743332, 38.0774727
25: -21.8450851, 20.7178764, -21.8450851, 20.7178764, -41.8004837, 41.8011856
26: -48.7503052, 23.4067230, -48.7503052, 23.4067230, -72.1570282, 72.1570282
27: -27.3533859, 21.0031128, -27.3533859, 21.0031128, -48.1854858, 48.1878052
28: -18.8090534, 22.7977848, -18.8090534, 22.7977848, -40.2042427, 40.2060127
29: -27.1733360, 20.9911938, -27.1733360, 20.9911938, -47.1224670, 47.1226044
30: -21.8117905, 18.9658546, -21.8117905, 18.9658546, -39.9446640, 39.9459572
31: -25.4966908, 18.5137157, -25.4966908, 18.5137157, -44.0104065, 44.0104065
32: -33.1184769, 16.2344227, -33.1184769, 16.2344227, -49.2141037, 49.2116470
33: -63.7556000, 12.8170786, -63.7556000, 12.8170786, -74.6335907, 74.6241455
34: -47.6960907, 8.2922297, -47.6960907, 8.2922297, -51.6944885, 51.6810608
35: -45.0071297, 11.5211735, -45.0071297, 11.5211735, -54.6324615, 54.6305542
36: -51.4329529, 13.5137806, -51.4329529, 13.5137806, -64.9467316, 64.9467316
37: -67.4206390, 1.4809723, -67.4206390, 1.4809723, -66.6803741, 66.6788635
38: -54.0977440, 16.3466244, -54.0977440, 16.3466244, -70.4443665, 70.4443665
39: -73.4575043, 1.7182684, -73.4575043, 1.7182684, -75.0498657, 75.0494690
40: -58.5168610, 0.1258335, -58.5168610, 0.1258335, -55.5898819, 55.5833740
41: -42.3647079, 12.1001272, -42.3647079, 12.1001272, -50.3980255, 50.3976212
42: -26.1969643, 15.6938057, -26.1969643, 15.6938057, -38.9777527, 38.9642868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=223, inp2_unstable=223, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1321
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 937
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1350
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1349

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4401281, upper bound: 33.4569306
time: 36.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.3993314, upper bound: 33.4953885
time: 38.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 77.91 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 77.91
Output dim: 11, lower bound: -33.4953885, upper bound: 33.3993314
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 77.91
Output dim: 11, lower bound: -33.4569306, upper bound: 33.4401281
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 77.91
Output dim: 11, lower bound: -33.4401281, upper bound: 33.4569306
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 77.91
Output dim: 11, lower bound: -33.3993314, upper bound: 33.4953885

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 52.38 + 233.87 = 286.25 seconds
