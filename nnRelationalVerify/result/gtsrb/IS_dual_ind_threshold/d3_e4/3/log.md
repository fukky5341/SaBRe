## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 7200 seconds
Split limit: 100
Threshold: 52.450938558


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402)
1: (-40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997)
2: (-37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302)
3: (-45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517)
4: (-53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861)
5: (-47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460)
6: (-68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607)
7: (-57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642)
8: (-47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555)
9: (-49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451)
10: (-79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202)
11: (-80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452)
12: (-74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725)
13: (-71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976)
14: (-107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280)
15: (-59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259)
16: (-83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727)
17: (-119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568)
18: (-69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162)
19: (-60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142)
20: (-54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261)
21: (-72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640)
22: (-82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304)
23: (-55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030)
24: (-64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188)
25: (-60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649)
26: (-93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533)
27: (-68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644)
28: (-56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477)
29: (-81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563)
30: (-68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482)
31: (-63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603)
32: (-65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784)
33: (-100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695)
34: (-85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313)
35: (-81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741)
36: (-82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160)
37: (-115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787)
38: (-102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727)
39: (-122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293)
40: (-97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832)
41: (-67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368)
42: (-49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.92 + 75.13 = 78.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -52.5034420, upper bound: 52.5034420

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4321920, upper bound: 52.4987991
time: 65.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4987990, upper bound: 52.4987991
time: 66.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 132.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 132.05
Output dim: 2, lower bound: -52.4321920, upper bound: 52.4987991
IS_A2, status: Status.UNKNOWN, split count: 1, time: 132.05
Output dim: 2, lower bound: -52.4987990, upper bound: 52.4987991

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -63.0016670, 46.1784630, -63.1355400, 46.2150993, -109.2167664, 109.3140030
1: -40.0608215, 41.9438629, -40.1383018, 41.9732971, -82.0341034, 82.0821686
2: -37.3429146, 43.9967194, -37.4892120, 44.0184250, -81.3613358, 81.4859314
3: -45.3487587, 52.1926460, -45.4825363, 52.2310410, -97.5798035, 97.6751862
4: -52.9470558, 40.6721992, -53.1258011, 40.7050781, -93.6521301, 93.7979889
5: -47.1742783, 57.2239685, -47.3120346, 57.2618027, -104.4360580, 104.5360031
6: -67.9469528, 41.8579330, -67.9994659, 41.9902153, -109.9371643, 109.8573914
7: -57.4120445, 53.1295624, -57.5179214, 53.1672325, -110.5792770, 110.6474838
8: -47.5625572, 47.2853432, -47.7240906, 47.3181801, -94.8807220, 95.0094299
9: -49.5510902, 52.8360023, -49.6013298, 52.9723740, -102.5234680, 102.4373322
10: -79.3242340, 77.1661530, -79.3909531, 77.4345856, -156.7588196, 156.5570984
11: -80.2767029, 53.3444786, -80.3379440, 53.5915794, -133.8682709, 133.6824188
12: -74.6675873, 59.2950630, -74.7130051, 59.6514626, -134.3190460, 134.0080719
13: -71.0004272, 66.5607605, -71.0461731, 66.7112427, -137.7116699, 137.6069336
14: -107.0077667, 57.4680252, -107.0892944, 57.6810379, -164.6887817, 164.5573120
15: -59.2739868, 50.7019501, -59.4334488, 50.7574577, -110.0314484, 110.1353989
16: -83.0029984, 66.6599503, -83.0863037, 66.8218384, -149.8248291, 149.7462463
17: -119.1782837, 79.0953064, -119.2388763, 79.4430771, -198.6213379, 198.3341827
18: -69.3224792, 42.3745041, -69.4155350, 42.4450264, -111.7674866, 111.7900391
19: -60.1685791, 25.1261559, -60.2202034, 25.1800632, -85.3486404, 85.3463593
20: -54.2787437, 32.4904060, -54.3253822, 32.5683022, -86.8470383, 86.8157883
21: -72.5295181, 36.9514503, -72.5829773, 37.0614853, -109.5909958, 109.5344238
22: -82.1464233, 48.2740059, -82.2346191, 48.3550415, -130.5014343, 130.5086212
23: -54.9721603, 34.8977737, -55.0195198, 34.9543839, -89.9265442, 89.9172897
24: -64.5143280, 34.7826767, -64.6462402, 34.8107872, -99.3251190, 99.4289169
25: -60.1498032, 39.7958908, -60.2208138, 39.8454590, -99.9952621, 100.0167084
26: -92.9877014, 51.0992432, -93.0506287, 51.2672806, -144.2549744, 144.1498413
27: -68.4123840, 44.3966713, -68.5595093, 44.4235687, -112.8359375, 112.9561768
28: -56.6728668, 36.6275101, -56.7205505, 36.6592445, -93.3321075, 93.3480606
29: -81.6733551, 54.4776573, -81.7365265, 54.5946503, -136.2679901, 136.2141724
30: -68.1156845, 37.1841164, -68.1694183, 37.2939987, -105.4096832, 105.3535309
31: -62.8120842, 30.8302631, -62.9132195, 30.8641911, -93.6762772, 93.7434769
32: -65.6613159, 48.1253128, -65.7137451, 48.2654037, -113.9267120, 113.8390579
33: -100.1179428, 58.5664062, -100.2993164, 58.6166229, -158.7345581, 158.8657227
34: -85.2263031, 44.6338501, -85.3240814, 44.6759109, -129.9022217, 129.9579163
35: -80.9530182, 47.4830894, -81.0908508, 47.5220757, -128.4750824, 128.5739441
36: -82.7201691, 48.5291252, -82.7809601, 48.5762863, -131.2964478, 131.3100891
37: -115.5018463, 48.2389297, -115.6158371, 48.2853394, -163.7871857, 163.8547668
38: -102.3634338, 63.6911469, -102.4558258, 63.7587929, -166.1222076, 166.1469727
39: -122.6005096, 54.8789597, -122.7386627, 54.9129181, -177.5134125, 177.6176147
40: -96.9164505, 47.6048622, -97.0514526, 47.6326675, -144.5491028, 144.6563110
41: -67.1906281, 40.1013680, -67.2579575, 40.1931190, -107.3837433, 107.3593216
42: -49.7730141, 45.0194016, -49.8178177, 45.2088432, -94.9818573, 94.8372116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 66.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
time: 66.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -63.3884659, 46.4092941, -63.3039589, 46.2607803, -109.6492462, 109.7132492
1: -40.2785645, 42.0561752, -40.2300758, 42.0080757, -82.2866364, 82.2862396
2: -37.7138939, 44.2348862, -37.6742897, 44.0449142, -81.7587967, 81.9091797
3: -45.6939659, 52.4097023, -45.6514282, 52.2787819, -97.9727478, 98.0611267
4: -53.4087982, 40.9603653, -53.3519592, 40.7447701, -94.1535645, 94.3123169
5: -47.5300598, 57.5281029, -47.4890060, 57.3090363, -104.8390961, 105.0171051
6: -68.2390747, 42.1814842, -68.0637665, 42.1372719, -110.3763428, 110.2452545
7: -57.7241211, 53.2586327, -57.6458664, 53.2081718, -110.9322891, 110.9044876
8: -47.9823532, 47.5534782, -47.9285774, 47.3577461, -95.3401031, 95.4820557
9: -49.7788239, 53.2098503, -49.6621666, 53.1442413, -102.9230652, 102.8720169
10: -79.8044739, 77.8485413, -79.4725952, 77.7758484, -157.5802917, 157.3211365
11: -80.6970978, 53.9490051, -80.4144592, 53.9060974, -134.6031952, 134.3634644
12: -75.2563324, 60.1641998, -74.7688446, 60.1077003, -135.3640289, 134.9330444
13: -71.2097778, 67.0046844, -71.1011505, 66.8995819, -138.1093292, 138.1058350
14: -107.4735489, 57.9991798, -107.1875839, 57.9612503, -165.4347839, 165.1867523
15: -59.6784286, 51.0344505, -59.6051788, 50.8272133, -110.5056458, 110.6396255
16: -83.3799133, 67.0744476, -83.1892166, 67.0164642, -150.3963776, 150.2636719
17: -119.6977081, 79.9486084, -119.3115158, 79.8942795, -199.5919647, 199.2601318
18: -69.6526642, 42.5886497, -69.5076752, 42.5296974, -112.1823578, 112.0963135
19: -60.4160500, 25.2779121, -60.2814751, 25.2472439, -85.6632919, 85.5593872
20: -54.5291977, 32.6928902, -54.3839569, 32.6659012, -87.1950989, 87.0768433
21: -72.8720551, 37.2309341, -72.6489716, 37.1999359, -110.0719833, 109.8799057
22: -82.3923416, 48.5531044, -82.3098602, 48.4566803, -130.8490295, 130.8629608
23: -55.1852112, 35.0653648, -55.0785713, 35.0247650, -90.2099686, 90.1439209
24: -64.8885498, 34.9694977, -64.8130188, 34.8434830, -99.7320251, 99.7825165
25: -60.3670120, 39.9802551, -60.2961884, 39.9066429, -100.2736511, 100.2764435
26: -93.2987366, 51.5121956, -93.1277618, 51.4626160, -144.7613525, 144.6399536
27: -68.8355484, 44.5341835, -68.7453461, 44.4545364, -113.2900620, 113.2795258
28: -56.8597832, 36.7313995, -56.7778206, 36.6957397, -93.5555115, 93.5092163
29: -81.8847046, 54.7821350, -81.7946396, 54.7379379, -136.6226501, 136.5767822
30: -68.3390198, 37.4886322, -68.2348938, 37.4333191, -105.7723389, 105.7235260
31: -63.1854401, 30.9387474, -63.0418549, 30.9034061, -94.0888443, 93.9805984
32: -65.9957962, 48.4745941, -65.7783203, 48.4427872, -114.4385757, 114.2529068
33: -100.6208496, 58.8414917, -100.5297165, 58.6769943, -159.2978516, 159.3712006
34: -85.5063934, 44.8319778, -85.4467773, 44.7243423, -130.2307434, 130.2787476
35: -81.3342209, 47.6985779, -81.2650757, 47.5701180, -128.9043427, 128.9636536
36: -82.9317322, 48.6838875, -82.8480682, 48.6309891, -131.5627136, 131.5319519
37: -115.8673401, 48.4071426, -115.7509384, 48.3421478, -164.2094727, 164.1580658
38: -102.6810760, 63.9082031, -102.5642929, 63.8408394, -166.5218811, 166.4725037
39: -123.0573502, 55.0508804, -122.9183121, 54.9520226, -178.0093536, 177.9691925
40: -97.3252945, 47.7326355, -97.2199860, 47.6584091, -144.9837036, 144.9526215
41: -67.4661102, 40.3551216, -67.3418045, 40.3094864, -107.7755890, 107.6969147
42: -50.0523376, 45.4968834, -49.8730850, 45.4501038, -95.5024414, 95.3699646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 680

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
time: 72.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 74.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 148.97 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 148.97
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 148.97
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 148.97
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 148.97
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -62.9879761, 46.1751900, -63.1106186, 46.2092896, -109.1972656, 109.2858124
1: -40.0516129, 41.9411545, -40.1218643, 41.9683571, -82.0199738, 82.0630188
2: -37.3303490, 43.9943962, -37.4663048, 44.0142365, -81.3445816, 81.4606934
3: -45.3344688, 52.1888542, -45.4563637, 52.2241745, -97.5586395, 97.6452179
4: -52.9326363, 40.6683044, -53.0998230, 40.6980438, -93.6306610, 93.7681274
5: -47.1603470, 57.2204399, -47.2863426, 57.2554855, -104.4158173, 104.5067825
6: -67.9405670, 41.8382339, -67.9880066, 41.9533653, -109.8939056, 109.8262405
7: -57.3995667, 53.1265984, -57.4950714, 53.1618462, -110.5614014, 110.6216736
8: -47.5489273, 47.2821083, -47.6994820, 47.3122902, -94.8612213, 94.9815903
9: -49.5459175, 52.8244705, -49.5920639, 52.9513206, -102.4972305, 102.4165344
10: -79.3172302, 77.1403122, -79.3782349, 77.3883133, -156.7055359, 156.5185394
11: -80.2709351, 53.3274994, -80.3278427, 53.5606575, -133.8315887, 133.6553345
12: -74.6628036, 59.2709084, -74.7043686, 59.6070328, -134.2698364, 133.9752655
13: -70.9860077, 66.5527573, -71.0202103, 66.6969223, -137.6829224, 137.5729675
14: -106.9992676, 57.4523582, -107.0737610, 57.6519623, -164.6512299, 164.5261230
15: -59.2522621, 50.6958351, -59.3939934, 50.7463379, -109.9985962, 110.0898285
16: -82.9942093, 66.6433716, -83.0705414, 66.7931061, -149.7873230, 149.7139130
17: -119.1724319, 79.0744171, -119.2280807, 79.4050140, -198.5774536, 198.3024902
18: -69.3159790, 42.3646164, -69.4037399, 42.4269600, -111.7429199, 111.7683411
19: -60.1643753, 25.1192265, -60.2126122, 25.1676331, -85.3320084, 85.3318329
20: -54.2742424, 32.4842072, -54.3172760, 32.5570679, -86.8313141, 86.8014832
21: -72.5241470, 36.9416733, -72.5733490, 37.0435944, -109.5677414, 109.5150223
22: -82.1398621, 48.2592812, -82.2226410, 48.3280182, -130.4678802, 130.4819183
23: -54.9680557, 34.8910446, -55.0120544, 34.9424667, -89.9105225, 89.9030838
24: -64.5047913, 34.7790298, -64.6294098, 34.8041840, -99.3089752, 99.4084396
25: -60.1446381, 39.7880898, -60.2111931, 39.8314514, -99.9760895, 99.9992676
26: -92.9820404, 51.0809631, -93.0405045, 51.2333488, -144.2153625, 144.1214600
27: -68.4010010, 44.3927078, -68.5391388, 44.4163361, -112.8173370, 112.9318466
28: -56.6689873, 36.6209831, -56.7136040, 36.6473312, -93.3163147, 93.3345795
29: -81.6671753, 54.4644852, -81.7255173, 54.5706711, -136.2378387, 136.1900024
30: -68.1107330, 37.1748505, -68.1606827, 37.2770538, -105.3877869, 105.3355331
31: -62.8041420, 30.8225746, -62.8983116, 30.8503609, -93.6544952, 93.7208862
32: -65.6549835, 48.1187515, -65.7024384, 48.2534065, -113.9083862, 113.8211823
33: -100.1052628, 58.5616531, -100.2758255, 58.6081543, -158.7134094, 158.8374786
34: -85.2173462, 44.6290512, -85.3075104, 44.6672134, -129.8845520, 129.9365540
35: -80.9410553, 47.4791794, -81.0691376, 47.5150452, -128.4561005, 128.5483093
36: -82.7114792, 48.5251961, -82.7653961, 48.5692062, -131.2806702, 131.2905884
37: -115.4909897, 48.2314987, -115.5960312, 48.2717361, -163.7627258, 163.8275146
38: -102.3515091, 63.6835632, -102.4343262, 63.7452126, -166.0967102, 166.1178894
39: -122.5873413, 54.8751411, -122.7145691, 54.9060898, -177.4934387, 177.5897064
40: -96.9053955, 47.6014671, -97.0314255, 47.6265106, -144.5319061, 144.6328888
41: -67.1839676, 40.0862656, -67.2458801, 40.1654053, -107.3493729, 107.3321457
42: -49.7677574, 45.0057945, -49.8082581, 45.1843605, -94.9521179, 94.8140411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4093444, upper bound: 52.4467989
time: 68.99 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
time: 72.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.0412064, 46.3334579, -62.6910744, 46.0331879, -109.0743790, 109.0245285
1: -40.0686798, 41.9912453, -39.8642235, 41.8649673, -81.9336319, 81.8554688
2: -37.3531342, 44.1866455, -37.0522499, 43.8648758, -81.2180023, 81.2388916
3: -45.3191147, 52.3325310, -45.0064507, 52.0428429, -97.3619537, 97.3389816
4: -53.0055542, 40.8882256, -52.6551056, 40.5511017, -93.5566559, 93.5433350
5: -47.1603127, 57.4587631, -46.8469849, 57.0562401, -104.2165527, 104.3057404
6: -68.1130753, 41.9654350, -67.8188400, 41.7516823, -109.8647461, 109.7842712
7: -57.4180527, 53.1905594, -57.1009560, 53.0429459, -110.4609909, 110.2915192
8: -47.6241379, 47.4846878, -47.3082657, 47.1607819, -94.7849197, 94.7929535
9: -49.6601868, 52.9059410, -49.4109192, 52.6111145, -102.2712860, 102.3168564
10: -79.6489182, 77.1517868, -79.0129700, 76.5853271, -156.2342529, 156.1647644
11: -80.5812683, 53.4360809, -80.1438141, 53.0422707, -133.6235352, 133.5798950
12: -75.1704483, 59.4216728, -74.3863220, 58.8322678, -134.0027161, 133.8079987
13: -71.0870972, 66.8132172, -70.8700409, 66.5335007, -137.6205750, 137.6832581
14: -107.2857132, 57.5401764, -106.7002182, 57.1662254, -164.4519348, 164.2403870
15: -59.3526154, 50.9105263, -59.0271759, 50.5836830, -109.9362946, 109.9376984
16: -83.2102203, 66.6921005, -82.8526154, 66.3538666, -149.5640869, 149.5447083
17: -119.5728226, 79.3014069, -118.9396133, 78.7840424, -198.3568573, 198.2410126
18: -69.5030060, 42.3487396, -69.1725464, 42.1083870, -111.6113892, 111.5212860
19: -60.3141441, 25.1132412, -60.0456505, 24.9653664, -85.2795105, 85.1588898
20: -54.4242210, 32.5242310, -54.1405678, 32.3750458, -86.7992706, 86.6647949
21: -72.7640991, 36.9553070, -72.3538055, 36.7235413, -109.4876175, 109.3091125
22: -82.2497711, 48.3115120, -82.0258026, 48.0240631, -130.2738342, 130.3373108
23: -55.0911865, 34.8973579, -54.8734207, 34.7313614, -89.8225479, 89.7707825
24: -64.7139130, 34.9049072, -64.4929962, 34.7353630, -99.4492798, 99.3979034
25: -60.2736816, 39.8347321, -60.1099815, 39.6429825, -99.9166489, 99.9446945
26: -93.1744690, 51.0151634, -92.7332916, 50.6016541, -143.7761230, 143.7484436
27: -68.5941010, 44.4727211, -68.3077393, 44.3370132, -112.9311142, 112.7804565
28: -56.7635498, 36.6486282, -56.5813332, 36.5398941, -93.3034363, 93.2299652
29: -81.7714005, 54.4639626, -81.5666733, 54.1945114, -135.9659119, 136.0306244
30: -68.2390060, 37.2344894, -68.0270309, 36.9901428, -105.2291336, 105.2615051
31: -63.0121422, 30.8118172, -62.6839027, 30.6829433, -93.6950836, 93.4957123
32: -65.8735504, 48.2450485, -65.5485535, 48.0474243, -113.9209595, 113.7936020
33: -100.2620239, 58.7297440, -99.9057159, 58.3938828, -158.6558838, 158.6354370
34: -85.2709198, 44.7331009, -85.0327148, 44.4985809, -129.7695007, 129.7658081
35: -81.0030899, 47.6096802, -80.6998978, 47.3444939, -128.3475800, 128.3095703
36: -82.7403870, 48.5983620, -82.5058441, 48.4575043, -131.1978912, 131.1042023
37: -115.6679840, 48.2774010, -115.3828506, 48.0931091, -163.7610779, 163.6602478
38: -102.4071503, 63.7939796, -102.0776062, 63.6136665, -166.0208130, 165.8715820
39: -122.7877197, 54.9675407, -122.4331436, 54.7700958, -177.5578003, 177.4006653
40: -97.0654449, 47.6772614, -96.7486267, 47.5148315, -144.5802765, 144.4258728
41: -67.3248291, 40.1838913, -67.0770416, 39.9984856, -107.3233032, 107.2609329
42: -49.9505310, 45.0964050, -49.6767540, 44.7616081, -94.7121429, 94.7731552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
time: 67.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753
time: 71.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 141.66 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 141.66
Output dim: 2, lower bound: -52.4093444, upper bound: 52.4467989
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 141.66
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 141.66
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 141.66
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -62.6348114, 46.0276108, -62.6099701, 45.8900909, -108.5249023, 108.6375809
1: -39.7979317, 41.7899704, -39.8176613, 41.7979927, -81.5959167, 81.6076355
2: -36.9824677, 43.9115753, -37.0032349, 43.7281151, -80.7105713, 80.9148102
3: -44.9119835, 51.9530411, -44.9524002, 51.8679199, -96.7798996, 96.9054413
4: -52.7027092, 40.7166519, -52.6015701, 40.4886246, -93.1913223, 93.3182144
5: -46.7241020, 57.0363808, -46.7873535, 56.8481674, -103.5722656, 103.8237305
6: -67.8604584, 41.5947456, -67.7546310, 41.6153183, -109.4757690, 109.3493729
7: -56.9550171, 52.7965622, -57.0258179, 52.8645439, -109.8195496, 109.8223801
8: -47.2673607, 47.1841927, -47.2647018, 47.0312233, -94.2985840, 94.4488907
9: -49.2896881, 52.4864426, -49.2493172, 52.5630417, -101.8527298, 101.7357635
10: -79.0531921, 76.4173889, -78.7358398, 76.5087662, -155.5619507, 155.1532135
11: -80.4021225, 53.0343018, -80.0385284, 52.9904175, -133.3925476, 133.0728302
12: -74.4197540, 58.5302086, -73.9974289, 58.7668304, -133.1865845, 132.5276337
13: -70.7096481, 66.3713837, -70.7132874, 66.4583435, -137.1679993, 137.0846558
14: -106.4499512, 56.8299828, -106.3194122, 57.1323013, -163.5822449, 163.1493988
15: -58.9287148, 50.6230888, -58.8975945, 50.5162544, -109.4449615, 109.5206833
16: -82.9266968, 66.2960815, -82.7400970, 66.2701263, -149.1968231, 149.0361786
17: -118.8831711, 78.4906616, -118.6067047, 78.7274399, -197.6106110, 197.0973663
18: -69.1384888, 42.0832214, -69.0488586, 42.0477867, -111.1862793, 111.1320648
19: -60.0811462, 24.9809818, -59.9617500, 24.9361534, -85.0172958, 84.9427338
20: -54.1795998, 32.3446732, -54.0507622, 32.3448792, -86.5244751, 86.3954315
21: -72.4870071, 36.6745110, -72.2465668, 36.6836853, -109.1706924, 108.9210815
22: -81.6915283, 47.9029465, -81.7794189, 47.9709816, -129.6625061, 129.6823730
23: -54.8354263, 34.7548294, -54.7963104, 34.6974945, -89.5329208, 89.5511398
24: -64.4113388, 34.7601280, -64.4295502, 34.6846542, -99.0959702, 99.1896744
25: -60.0073242, 39.5848656, -60.0120888, 39.5875511, -99.5948792, 99.5969543
26: -92.3921509, 50.2813911, -92.3560791, 50.5351372, -142.9272919, 142.6374664
27: -68.2030640, 44.3067818, -68.2386246, 44.2740974, -112.4771423, 112.5454025
28: -56.5153847, 36.5370026, -56.5143242, 36.4989929, -93.0143738, 93.0513229
29: -81.3581772, 53.9963455, -81.3910828, 54.1525154, -135.5106964, 135.3874207
30: -67.9536514, 36.9807587, -67.9564285, 36.9436378, -104.8972931, 104.9371872
31: -62.6892624, 30.6513195, -62.6017532, 30.6368675, -93.3261261, 93.2530746
32: -65.6617661, 47.9892807, -65.4777908, 47.9988632, -113.6606216, 113.4670639
33: -99.8089600, 58.5538864, -99.8320541, 58.3144569, -158.1234131, 158.3859406
34: -84.8668671, 44.5033035, -84.9666290, 44.4032745, -129.2701263, 129.4699097
35: -80.5944061, 47.4234238, -80.6368561, 47.2688751, -127.8632812, 128.0602722
36: -82.4383469, 48.4553146, -82.4319916, 48.3995972, -130.8379211, 130.8872986
37: -115.2853928, 48.0735168, -115.2863083, 48.0289459, -163.3143311, 163.3598175
38: -101.9582443, 63.5327187, -102.0028763, 63.5214691, -165.4797058, 165.5355835
39: -122.4164429, 54.7245483, -122.3497467, 54.6871872, -177.1036377, 177.0742950
40: -96.6351776, 47.3770676, -96.6851654, 47.3646507, -143.9998169, 144.0622253
41: -67.0232391, 39.9061127, -67.0234985, 39.8998604, -106.9230957, 106.9296036
42: -49.7662773, 44.7607574, -49.6124039, 44.6883278, -94.4545975, 94.3731613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3081349
time: 61.10 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
time: 70.68 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -63.0062828, 46.3049927, -62.6697006, 46.0156403, -109.0219116, 108.9746933
1: -40.0494308, 41.9658737, -39.8524933, 41.8488388, -81.8982697, 81.8183670
2: -37.3254166, 44.1667938, -37.0356331, 43.8529091, -81.1783295, 81.2024155
3: -45.2934074, 52.2674217, -44.9908829, 52.0060654, -97.2994537, 97.2583008
4: -52.9699898, 40.8711128, -52.6337967, 40.5406761, -93.5106506, 93.5048981
5: -47.1376038, 57.3972015, -46.8330231, 57.0233345, -104.1609192, 104.2302170
6: -68.0921173, 41.8563995, -67.8061142, 41.6855469, -109.7776489, 109.6625137
7: -57.3941727, 53.1208382, -57.0865211, 52.9967766, -110.3909302, 110.2073517
8: -47.5920143, 47.4658813, -47.2889862, 47.1493454, -94.7413635, 94.7548676
9: -49.6314240, 52.8838043, -49.3932571, 52.5979767, -102.2294006, 102.2770615
10: -79.6148529, 77.1051025, -78.9922485, 76.5573273, -156.1721802, 156.0973511
11: -80.5539551, 53.3897018, -80.1274414, 53.0146065, -133.5685577, 133.5171509
12: -75.1300278, 59.3751831, -74.3619690, 58.8044434, -133.9344788, 133.7371521
13: -71.0519867, 66.7773819, -70.8484650, 66.5121994, -137.5641785, 137.6258240
14: -107.2305069, 57.5241394, -106.6662140, 57.1566010, -164.3871155, 164.1903534
15: -59.2113266, 50.8908195, -58.9467621, 50.5718422, -109.7831726, 109.8375778
16: -83.1816635, 66.6532364, -82.8352585, 66.3308411, -149.5125122, 149.4884949
17: -119.5276718, 79.2307816, -118.9119034, 78.7422333, -198.2698975, 198.1426849
18: -69.4657822, 42.3213043, -69.1501312, 42.0917320, -111.5575104, 111.4714355
19: -60.2923851, 25.0960903, -60.0324059, 24.9550514, -85.2474365, 85.1284790
20: -54.4038162, 32.5086594, -54.1281128, 32.3655968, -86.7694092, 86.6367722
21: -72.7362976, 36.9390030, -72.3365631, 36.7136002, -109.4498901, 109.2755585
22: -82.1423569, 48.2908096, -81.9581909, 48.0117531, -130.1541138, 130.2489929
23: -55.0741539, 34.8796234, -54.8629608, 34.7206154, -89.7947617, 89.7425842
24: -64.6868286, 34.8896675, -64.4768066, 34.7259178, -99.4127350, 99.3664627
25: -60.2199860, 39.8187943, -60.0779686, 39.6333389, -99.8533173, 99.8967438
26: -93.0828323, 50.9904861, -92.6736450, 50.5866928, -143.6695251, 143.6641235
27: -68.5572052, 44.4516602, -68.2856827, 44.3233719, -112.8805771, 112.7373428
28: -56.7471695, 36.6335602, -56.5712013, 36.5307350, -93.2779083, 93.2047577
29: -81.7256470, 54.4450111, -81.5374680, 54.1832123, -135.9088440, 135.9824829
30: -68.2150269, 37.2061996, -68.0125809, 36.9730911, -105.1881180, 105.2187729
31: -62.9850235, 30.7924900, -62.6673431, 30.6712570, -93.6562729, 93.4598312
32: -65.8507538, 48.2169342, -65.5346985, 48.0306664, -113.8814163, 113.7516251
33: -100.2277908, 58.7068520, -99.8851776, 58.3798523, -158.6076355, 158.5920258
34: -85.2455292, 44.7075577, -85.0172501, 44.4826202, -129.7281494, 129.7248077
35: -80.9673004, 47.5904427, -80.6783447, 47.3327408, -128.3000488, 128.2687836
36: -82.7142029, 48.5831146, -82.4899673, 48.4481239, -131.1623230, 131.0730896
37: -115.6252594, 48.2589645, -115.3573990, 48.0818024, -163.7070465, 163.6163635
38: -102.3765945, 63.7689743, -102.0593033, 63.5984344, -165.9750214, 165.8282776
39: -122.7313690, 54.9483719, -122.3998795, 54.7580986, -177.4894562, 177.3482361
40: -97.0370560, 47.6510887, -96.7316132, 47.4981728, -144.5352325, 144.3827057
41: -67.3062363, 40.1226044, -67.0658112, 39.9605484, -107.2667847, 107.1884155
42: -49.9310303, 45.0474548, -49.6649742, 44.7325134, -94.6635437, 94.7124176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
time: 71.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524130
time: 115.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 189.11 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 189.11
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3081349
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 189.11
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 189.11
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 189.11
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524130

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -62.3622818, 45.9521866, -62.1636086, 45.7662888, -108.1285629, 108.1157990
1: -39.6432953, 41.7241287, -39.5667953, 41.6896286, -81.3329239, 81.2909164
2: -36.7288551, 43.8592606, -36.5891685, 43.6423492, -80.3712006, 80.4484177
3: -44.6183281, 51.8778000, -44.4704781, 51.7431259, -96.3614502, 96.3482819
4: -52.3989792, 40.6498795, -52.1100845, 40.3791275, -92.7781067, 92.7599640
5: -46.4584923, 56.9667702, -46.3513794, 56.7334480, -103.1919403, 103.3181458
6: -67.7442474, 41.4548187, -67.5644989, 41.3911781, -109.1354218, 109.0193100
7: -56.7531815, 52.7334366, -56.6930199, 52.7615242, -109.5147095, 109.4264526
8: -47.0462151, 47.1085510, -46.9063263, 46.9071312, -93.9533463, 94.0148773
9: -49.1330490, 52.3285255, -48.9934692, 52.3110237, -101.4440765, 101.3219833
10: -78.9165192, 75.8574371, -78.5114441, 75.5987625, -154.5152893, 154.3688812
11: -80.2922974, 52.5227737, -79.8533630, 52.1609383, -132.4532318, 132.3761292
12: -74.3341522, 58.0239868, -73.8568115, 57.9354439, -132.2695923, 131.8807983
13: -70.4571838, 66.2160492, -70.3034210, 66.2057343, -136.6629181, 136.5194702
14: -106.2711258, 56.4187431, -106.0256119, 56.4550018, -162.7261353, 162.4443512
15: -58.6613388, 50.5096474, -58.4694328, 50.3294144, -108.9907532, 108.9790802
16: -82.7534027, 66.0148773, -82.4549866, 65.8323669, -148.5857544, 148.4698486
17: -118.7519913, 77.8624420, -118.3914413, 77.6932068, -196.4451904, 196.2538757
18: -68.9920197, 41.8001404, -68.8080521, 41.5825729, -110.5745850, 110.6081924
19: -59.9847832, 24.7635536, -59.8046761, 24.5808544, -84.5656357, 84.5682297
20: -54.0834923, 32.1680069, -53.8916206, 32.0541039, -86.1375961, 86.0596313
21: -72.3796234, 36.3597260, -72.0710373, 36.1651306, -108.5447540, 108.4307556
22: -81.5690460, 47.6649399, -81.5828247, 47.5795021, -129.1485291, 129.2477722
23: -54.7507095, 34.5343628, -54.6568184, 34.3370438, -89.0877533, 89.1911774
24: -64.3015823, 34.6475372, -64.2524567, 34.4955597, -98.7971420, 98.8999863
25: -59.9157295, 39.4200134, -59.8633881, 39.3165550, -99.2322845, 99.2834015
26: -92.2762756, 49.8795204, -92.1658859, 49.8692169, -142.1454773, 142.0454102
27: -68.0359039, 44.1782875, -67.9687195, 44.0660172, -112.1019211, 112.1470032
28: -56.4270363, 36.4250908, -56.3704758, 36.3178444, -92.7448730, 92.7955627
29: -81.2608414, 53.6708488, -81.2341461, 53.6168022, -134.8776398, 134.9049988
30: -67.8552094, 36.7349205, -67.7943726, 36.5428238, -104.3980179, 104.5292969
31: -62.5450516, 30.4426041, -62.3612633, 30.2953758, -92.8404236, 92.8038559
32: -65.5334930, 47.8501015, -65.2659378, 47.7760468, -113.3095398, 113.1160355
33: -99.4379349, 58.4601135, -99.2242508, 58.1599464, -157.5978699, 157.6843567
34: -84.6203918, 44.4132500, -84.5601196, 44.2575951, -128.8779755, 128.9733582
35: -80.2501221, 47.3444214, -80.0707626, 47.1411285, -127.3912506, 127.4151840
36: -82.1774902, 48.3837509, -82.0073395, 48.2822266, -130.4597168, 130.3910828
37: -115.0835876, 47.9614334, -114.9608231, 47.8473663, -162.9309540, 162.9222565
38: -101.7036972, 63.4332428, -101.5899811, 63.3602829, -165.0639801, 165.0232239
39: -122.0812836, 54.6518936, -121.8016357, 54.5693512, -176.6506348, 176.4535217
40: -96.4005280, 47.3278122, -96.3002396, 47.2836227, -143.6841431, 143.6280518
41: -66.8971176, 39.7735443, -66.8163910, 39.6882706, -106.5853882, 106.5899353
42: -49.6744804, 44.4548492, -49.4594574, 44.1926003, -93.8670807, 93.9143066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
time: 61.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 70.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -62.6024971, 46.0168076, -62.6595383, 46.0833664, -108.6858521, 108.6763458
1: -39.7784538, 41.7795334, -39.8379364, 41.8562088, -81.6346588, 81.6174698
2: -36.9574356, 43.9038315, -37.0057220, 43.8825684, -80.8400040, 80.9095535
3: -44.8800278, 51.9403496, -44.9455338, 52.1051292, -96.9851456, 96.8858795
4: -52.6692085, 40.7051544, -52.6088562, 40.5885239, -93.2577286, 93.3140030
5: -46.6952896, 57.0248833, -46.7891426, 57.0678825, -103.7631683, 103.8140259
6: -67.8415833, 41.5401688, -67.7927551, 41.6062813, -109.4478607, 109.3329239
7: -56.9298592, 52.7825165, -57.0514107, 52.9116936, -109.8415527, 109.8339233
8: -47.2411499, 47.1722260, -47.2732239, 47.1211662, -94.3623047, 94.4454422
9: -49.2573547, 52.4595108, -49.2726555, 52.5866165, -101.8439713, 101.7321625
10: -79.0359497, 76.3556976, -79.0692978, 76.4961700, -155.5321198, 155.4249878
11: -80.3865356, 52.9939461, -80.3744965, 52.9604683, -133.3470001, 133.3684387
12: -74.4044342, 58.4826965, -74.3640900, 58.7540894, -133.1585236, 132.8467712
13: -70.6695862, 66.3495636, -70.7274323, 66.6687012, -137.3382874, 137.0769806
14: -106.4237518, 56.7881470, -106.6476517, 57.1080894, -163.5318451, 163.4357910
15: -58.8681717, 50.6065636, -58.9054222, 50.5499802, -109.4181519, 109.5119781
16: -82.9031677, 66.2453766, -82.8316498, 66.2701874, -149.1733551, 149.0770264
17: -118.8655319, 78.4292450, -119.0025177, 78.6891937, -197.5547180, 197.4317627
18: -69.1166763, 42.0523834, -69.3451080, 42.0488892, -111.1655426, 111.3974915
19: -60.0690765, 24.9572411, -60.1693153, 24.9259529, -84.9950180, 85.1265564
20: -54.1656342, 32.3276062, -54.2428551, 32.3517609, -86.5173950, 86.5704651
21: -72.4706650, 36.6438065, -72.5415955, 36.6734085, -109.1440582, 109.1854019
22: -81.6648788, 47.8750763, -81.8515549, 47.9767914, -129.6416626, 129.7266235
23: -54.8238640, 34.7309265, -54.9963951, 34.7005730, -89.5244293, 89.7273254
24: -64.3921356, 34.7447701, -64.5439072, 34.6914444, -99.0835724, 99.2886810
25: -59.9937172, 39.5628166, -60.1143265, 39.5907288, -99.5844421, 99.6771393
26: -92.3712769, 50.2410965, -92.7449112, 50.5427513, -142.9140167, 142.9860077
27: -68.1771164, 44.2835236, -68.2941589, 44.2810593, -112.4581757, 112.5776825
28: -56.5043755, 36.5207977, -56.6435699, 36.5229378, -93.0273132, 93.1643677
29: -81.3373337, 53.9593048, -81.4818344, 54.1345596, -135.4718781, 135.4411316
30: -67.9389954, 36.9567757, -68.1853333, 36.9654961, -104.9044952, 105.1421051
31: -62.6717339, 30.6274567, -62.8209076, 30.6231079, -93.2948456, 93.4483643
32: -65.6406174, 47.9668961, -65.5109100, 48.0173111, -113.6579285, 113.4778061
33: -99.7749023, 58.5412979, -99.8467560, 58.5994530, -158.3743591, 158.3880615
34: -84.8369751, 44.4901505, -84.9799271, 44.5240822, -129.3610535, 129.4700623
35: -80.5563049, 47.4139366, -80.6329193, 47.4867554, -128.0430603, 128.0468597
36: -82.4048309, 48.4453659, -82.4288635, 48.4825134, -130.8873444, 130.8742371
37: -115.2484894, 48.0592613, -115.3378677, 48.0798721, -163.3283691, 163.3971252
38: -101.9159088, 63.5198135, -102.0290527, 63.6198006, -165.5357056, 165.5488586
39: -122.3709412, 54.7143898, -122.3675385, 54.9044304, -177.2753754, 177.0819244
40: -96.6022263, 47.3681145, -96.7267532, 47.4895287, -144.0917511, 144.0948639
41: -67.0041046, 39.8694229, -67.0601501, 39.9275665, -106.9316711, 106.9295731
42: -49.7518158, 44.7196350, -49.6568794, 44.7050552, -94.4568634, 94.3765106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
time: 75.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
time: 71.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.7333069, 46.2300644, -62.2233543, 45.8917999, -108.6251068, 108.4534149
1: -39.8944168, 41.8999672, -39.6016083, 41.7398834, -81.6342850, 81.5015717
2: -37.0713882, 44.1146889, -36.6214600, 43.7673340, -80.8387146, 80.7361450
3: -44.9986115, 52.1924667, -44.5092278, 51.8822708, -96.8808823, 96.7016907
4: -52.6658440, 40.8039627, -52.1422691, 40.4307518, -93.0965958, 92.9462280
5: -46.8712006, 57.3278503, -46.3970566, 56.9085159, -103.7797165, 103.7248993
6: -67.9754105, 41.7163010, -67.6153564, 41.4615593, -109.4369659, 109.3316498
7: -57.1902122, 53.0583038, -56.7539253, 52.8936043, -110.0838165, 109.8122253
8: -47.3705330, 47.3904724, -46.9306107, 47.0253143, -94.3958282, 94.3210831
9: -49.4751434, 52.7252350, -49.1377068, 52.3459816, -101.8211212, 101.8629456
10: -79.4791031, 76.5446472, -78.7683182, 75.6472931, -155.1264038, 155.3129578
11: -80.4410248, 52.8785896, -79.9414978, 52.1845398, -132.6255493, 132.8200836
12: -75.0448151, 58.8685455, -74.2214966, 57.9729691, -133.0177917, 133.0900421
13: -70.8004379, 66.6211624, -70.4391403, 66.2584686, -137.0588989, 137.0603027
14: -107.0525055, 57.1126671, -106.3725662, 56.4793625, -163.5318604, 163.4852295
15: -58.9482994, 50.7767792, -58.5256920, 50.3844299, -109.3327103, 109.3024750
16: -83.0073090, 66.3704987, -82.5495758, 65.8938904, -148.9011993, 148.9200745
17: -119.3969116, 78.6023331, -118.6968536, 77.7076111, -197.1044922, 197.2991791
18: -69.3179321, 42.0379219, -68.9088440, 41.6265182, -110.9444427, 110.9467621
19: -60.1960258, 24.8786449, -59.8751259, 24.5995579, -84.7955780, 84.7537689
20: -54.3079948, 32.3316841, -53.9682388, 32.0747643, -86.3827515, 86.2999268
21: -72.6292419, 36.6239014, -72.1602020, 36.1949844, -108.8242188, 108.7841034
22: -82.0207520, 48.0518303, -81.7621231, 47.6202621, -129.6410217, 129.8139496
23: -54.9887390, 34.6589966, -54.7226906, 34.3601532, -89.3488922, 89.3816833
24: -64.5761032, 34.7764969, -64.2991943, 34.5367203, -99.1128235, 99.0756836
25: -60.1283989, 39.6537781, -59.9290161, 39.3623238, -99.4907227, 99.5827866
26: -92.9675217, 50.5864449, -92.4851608, 49.9207726, -142.8882904, 143.0716095
27: -68.3891754, 44.3234291, -68.0154419, 44.1159019, -112.5050812, 112.3388672
28: -56.6581993, 36.5214882, -56.4268761, 36.3493423, -93.0075378, 92.9483643
29: -81.6289978, 54.1187973, -81.3811646, 53.6474609, -135.2764587, 135.4999390
30: -68.1160049, 36.9601059, -67.8500061, 36.5722351, -104.6882401, 104.8101120
31: -62.8381310, 30.5837288, -62.4266472, 30.3295288, -93.1676636, 93.0103760
32: -65.7220612, 48.0772324, -65.3223114, 47.8074875, -113.5295486, 113.3995361
33: -99.8564835, 58.6136017, -99.2771683, 58.2245750, -158.0810547, 157.8907623
34: -84.9983978, 44.6178856, -84.6105728, 44.3364105, -129.3348083, 129.2284546
35: -80.6226501, 47.5123100, -80.1120453, 47.2039948, -127.8266449, 127.6243591
36: -82.4524078, 48.5103836, -82.0655136, 48.3299942, -130.7824097, 130.5758972
37: -115.4230728, 48.1465607, -115.0314102, 47.8996887, -163.3227539, 163.1779785
38: -102.1209564, 63.6693268, -101.6459732, 63.4367447, -165.5576935, 165.3153076
39: -122.3959961, 54.8753357, -121.8509598, 54.6396103, -177.0356140, 176.7262878
40: -96.8021240, 47.6015663, -96.3469162, 47.4171486, -144.2192688, 143.9484863
41: -67.1791534, 39.9905586, -66.8584900, 39.7499199, -106.9290771, 106.8490448
42: -49.8374138, 44.7407074, -49.5114250, 44.2370224, -94.0744324, 94.2521286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 69.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 70.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.9739151, 46.2942352, -62.7190514, 46.2089157, -109.1828308, 109.0132751
1: -40.0298767, 41.9553757, -39.8725128, 41.9068489, -81.9367218, 81.8278885
2: -37.3003922, 44.1589813, -37.0377769, 44.0074615, -81.3078537, 81.1967621
3: -45.2613106, 52.2546463, -44.9839096, 52.2437630, -97.5050735, 97.2385406
4: -52.9364166, 40.8595619, -52.6407394, 40.6404037, -93.5768204, 93.5002899
5: -47.1087189, 57.3855934, -46.8345337, 57.2428398, -104.3515625, 104.2201233
6: -68.0732117, 41.8020554, -67.8437195, 41.6760139, -109.7492218, 109.6457596
7: -57.3687592, 53.1069183, -57.1116905, 53.0439911, -110.4127502, 110.2186127
8: -47.5658379, 47.4538498, -47.2972794, 47.2393799, -94.8052216, 94.7511292
9: -49.5991173, 52.8569031, -49.4161949, 52.6215515, -102.2206726, 102.2731018
10: -79.5976715, 77.0433655, -79.3261108, 76.5443649, -156.1420288, 156.3694763
11: -80.5379181, 53.3493156, -80.4632416, 52.9842682, -133.5221863, 133.8125610
12: -75.1146088, 59.3276443, -74.7286911, 58.7914352, -133.9060364, 134.0563354
13: -71.0116577, 66.7554932, -70.8621292, 66.7228928, -137.7345581, 137.6176147
14: -107.2041245, 57.4823456, -106.9950180, 57.1322784, -164.3363953, 164.4773560
15: -59.1511116, 50.8741989, -58.9574509, 50.6053810, -109.7564926, 109.8316498
16: -83.1577301, 66.6032104, -82.9266205, 66.3305817, -149.4883118, 149.5298157
17: -119.5099030, 79.1694794, -119.3079453, 78.7036896, -198.2135925, 198.4774170
18: -69.4436340, 42.2903976, -69.4471893, 42.0925827, -111.5362091, 111.7375641
19: -60.2802544, 25.0723610, -60.2400169, 24.9447384, -85.2249756, 85.3123779
20: -54.3897858, 32.4915161, -54.3204498, 32.3722649, -86.7620544, 86.8119583
21: -72.7199249, 36.9082527, -72.6312714, 36.7032166, -109.4231415, 109.5395203
22: -82.1160126, 48.2628365, -82.0308762, 48.0171661, -130.1331787, 130.2937012
23: -55.0625000, 34.8557396, -55.0629768, 34.7235489, -89.7860413, 89.9187164
24: -64.6675797, 34.8742447, -64.5921021, 34.7324715, -99.4000549, 99.4663467
25: -60.2062645, 39.7967033, -60.1801758, 39.6362495, -99.8425140, 99.9768829
26: -93.0616913, 50.9499321, -93.0632172, 50.5940323, -143.6557312, 144.0131531
27: -68.5312042, 44.4286270, -68.3418961, 44.3300552, -112.8612595, 112.7705231
28: -56.7360687, 36.6173286, -56.7003479, 36.5542717, -93.2903290, 93.3176727
29: -81.7047424, 54.4079895, -81.6286392, 54.1649628, -135.8697052, 136.0366211
30: -68.2003326, 37.1820526, -68.2413635, 36.9947281, -105.1950607, 105.4234161
31: -62.9671402, 30.7685947, -62.8872223, 30.6571960, -93.6243286, 93.6558151
32: -65.8295593, 48.1945953, -65.5672913, 48.0488472, -113.8784027, 113.7618866
33: -100.1937027, 58.6942711, -99.8996735, 58.6644249, -158.8581085, 158.5939331
34: -85.2156219, 44.6944008, -85.0304794, 44.6032906, -129.8188934, 129.7248840
35: -80.9290771, 47.5809937, -80.6741638, 47.5503120, -128.4793854, 128.2551575
36: -82.6805725, 48.5729370, -82.4865875, 48.5312462, -131.2118225, 131.0595245
37: -115.5884781, 48.2446823, -115.4084930, 48.1323051, -163.7207794, 163.6531677
38: -102.3345184, 63.7559738, -102.0850143, 63.6964226, -166.0309296, 165.8409882
39: -122.6859436, 54.9381256, -122.4167557, 54.9753227, -177.6612701, 177.3548737
40: -97.0040359, 47.6418839, -96.7730865, 47.6236877, -144.6277161, 144.4149780
41: -67.2869949, 40.0861092, -67.1021576, 39.9883308, -107.2753220, 107.1882629
42: -49.9164581, 45.0063057, -49.7093773, 44.7491341, -94.6655884, 94.7156830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1503
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 680

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
time: 71.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 158.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 232.86 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 232.86
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 78.05 + 1684.13 = 1762.19 seconds
