## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 3.01 + 79.71 = 82.72 seconds
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4321920, upper bound: 52.4987991
time: 69.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4987990, upper bound: 52.4987991
time: 70.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 140.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 140.30
Output dim: 2, lower bound: -52.4321920, upper bound: 52.4987991
IS_A2, status: Status.UNKNOWN, split count: 1, time: 140.30
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

Time for backsubstitution: 2.37 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 70.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
time: 70.37 seconds

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

Time for backsubstitution: 2.38 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
time: 76.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 79.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 157.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 157.83
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 157.83
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 157.83
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 157.83
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -62.6544685, 46.1017761, -62.5230408, 45.9868240, -108.6412964, 108.6248169
1: -39.8505402, 41.8790665, -39.7722549, 41.8298264, -81.6803665, 81.6513214
2: -36.9819603, 43.9480438, -36.8671417, 43.8380508, -80.8200073, 80.8151855
3: -44.9740677, 52.1148300, -44.8377838, 51.9942856, -96.9683533, 96.9525986
4: -52.5435448, 40.5996666, -52.4288750, 40.5111771, -93.0547104, 93.0285416
5: -46.8044968, 57.1540260, -46.6701469, 57.0084457, -103.8129425, 103.8241577
6: -67.8200531, 41.6456909, -67.7539825, 41.6062927, -109.4263458, 109.3996582
7: -57.1067657, 53.0615883, -56.9729843, 53.0014267, -110.1081848, 110.0345764
8: -47.2042656, 47.2160072, -47.1039124, 47.1208458, -94.3251114, 94.3199158
9: -49.4308510, 52.5322227, -49.3489075, 52.4396591, -101.8704987, 101.8811264
10: -79.1669617, 76.4689941, -78.9302979, 76.2438660, -155.4108276, 155.3992920
11: -80.1586838, 52.8314743, -80.0663910, 52.7279129, -132.8865967, 132.8978577
12: -74.5809631, 58.5524979, -74.3299026, 58.3761978, -132.9571533, 132.8824005
13: -70.8768616, 66.3708267, -70.8146057, 66.3450394, -137.2218933, 137.1854248
14: -106.8188705, 57.0098686, -106.6009216, 56.8870621, -163.7059326, 163.6107941
15: -58.9600105, 50.5771980, -58.8678131, 50.5132065, -109.4732208, 109.4450073
16: -82.8319321, 66.2839966, -82.7475281, 66.1643982, -148.9963379, 149.0315247
17: -119.0527267, 78.4491882, -118.8663025, 78.3338013, -197.3865356, 197.3154907
18: -69.1740417, 42.1359978, -69.0801086, 42.0256119, -111.1996536, 111.2161102
19: -60.0670586, 24.9610500, -59.9835014, 24.8980656, -84.9651260, 84.9445496
20: -54.1729546, 32.3217812, -54.0811157, 32.2775955, -86.4505463, 86.4028931
21: -72.4206085, 36.6754990, -72.2869873, 36.5849266, -109.0055237, 108.9624710
22: -82.0042953, 48.0309029, -81.9504395, 47.9220772, -129.9263763, 129.9813232
23: -54.8777809, 34.7294846, -54.8139191, 34.6607704, -89.5385437, 89.5434036
24: -64.3406754, 34.7171707, -64.3279877, 34.7023582, -99.0430298, 99.0451508
25: -60.0566673, 39.6492233, -60.0345650, 39.5817947, -99.6384583, 99.6837769
26: -92.8636627, 50.6022720, -92.6554489, 50.4062805, -143.2699432, 143.2577209
27: -68.1719360, 44.3345718, -68.1232605, 44.3059082, -112.4778290, 112.4578323
28: -56.5768814, 36.5446663, -56.5241165, 36.5032349, -93.0801163, 93.0687866
29: -81.5604477, 54.1588669, -81.5087967, 54.0512733, -135.6117249, 135.6676636
30: -68.0159454, 36.9302101, -67.9611588, 36.8510017, -104.8669357, 104.8913651
31: -62.6415710, 30.7030678, -62.5577812, 30.6436424, -93.2852173, 93.2608490
32: -65.5369568, 47.8959351, -65.4834747, 47.8702927, -113.4072495, 113.3794022
33: -99.7595444, 58.4539146, -99.6757965, 58.3323059, -158.0918427, 158.1296997
34: -84.9915314, 44.5345726, -84.9106293, 44.4494781, -129.4410095, 129.4452057
35: -80.6222534, 47.3930893, -80.5260315, 47.2954712, -127.9177246, 127.9191208
36: -82.5289612, 48.4438248, -82.4387817, 48.4020653, -130.9310303, 130.8825989
37: -115.3033447, 48.1088753, -115.2481918, 48.0359344, -163.3392792, 163.3570709
38: -102.0901794, 63.5780754, -101.9697037, 63.5330238, -165.6231842, 165.5477753
39: -122.3325806, 54.7955933, -122.2549591, 54.7305946, -177.0631714, 177.0505371
40: -96.6571579, 47.5497055, -96.5809402, 47.4884186, -144.1455536, 144.1306458
41: -67.0490341, 39.9345932, -66.9928436, 39.8848381, -106.9338684, 106.9274216
42: -49.6698685, 44.6195183, -49.6210556, 44.5208473, -94.1907196, 94.2405701

Time for backsubstitution: 2.36 seconds

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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 679
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 824
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
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
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
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3590838
time: 65.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
time: 69.85 seconds

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

Time for backsubstitution: 2.37 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4265702, upper bound: 52.4467989
time: 102.21 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
time: 76.61 seconds

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

Time for backsubstitution: 2.38 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
time: 71.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753
time: 75.95 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.3747635, 46.4060402, -63.2789803, 46.2549477, -109.6296997, 109.6850128
1: -40.2693939, 42.0534286, -40.2135582, 42.0030746, -82.2724609, 82.2669830
2: -37.7013588, 44.2325668, -37.6513557, 44.0406952, -81.7420502, 81.8839264
3: -45.6796799, 52.4059410, -45.6251488, 52.2718658, -97.9515457, 98.0310898
4: -53.3943901, 40.9564972, -53.3259087, 40.7377167, -94.1321106, 94.2824020
5: -47.5161438, 57.5245819, -47.4632416, 57.3026581, -104.8188019, 104.9878006
6: -68.2327728, 42.1613617, -68.0522614, 42.0998878, -110.3326569, 110.2136154
7: -57.7115898, 53.2556572, -57.6229973, 53.2027435, -110.9143372, 110.8786545
8: -47.9687462, 47.5502548, -47.9038849, 47.3518219, -95.3205719, 95.4541397
9: -49.7737045, 53.1983948, -49.6528816, 53.1231880, -102.8968964, 102.8512726
10: -79.7975159, 77.8227463, -79.4598541, 77.7294922, -157.5270081, 157.2825928
11: -80.6915131, 53.9320068, -80.4043427, 53.8750267, -134.5665436, 134.3363342
12: -75.2515869, 60.1400986, -74.7601471, 60.0632362, -135.3148193, 134.9002380
13: -71.1954041, 66.9966736, -71.0750961, 66.8852005, -138.0805969, 138.0717773
14: -107.4651031, 57.9834938, -107.1720276, 57.9320335, -165.3971100, 165.1555176
15: -59.6580124, 51.0283585, -59.5677109, 50.8160744, -110.4740906, 110.5960617
16: -83.3711548, 67.0578156, -83.1734161, 66.9890594, -150.3602142, 150.2312317
17: -119.6918411, 79.9277420, -119.3007202, 79.8560867, -199.5479126, 199.2284546
18: -69.6460724, 42.5787163, -69.4958496, 42.5115128, -112.1575851, 112.0745621
19: -60.4118423, 25.2710152, -60.2738724, 25.2347946, -85.6466370, 85.5448914
20: -54.5247612, 32.6866913, -54.3758583, 32.6545868, -87.1793365, 87.0625458
21: -72.8667755, 37.2211990, -72.6393585, 37.1819649, -110.0487366, 109.8605423
22: -82.3854980, 48.5384407, -82.2974472, 48.4295616, -130.8150635, 130.8358917
23: -55.1811562, 35.0586166, -55.0710907, 35.0128326, -90.1939774, 90.1297073
24: -64.8789978, 34.9658585, -64.7964325, 34.8368454, -99.7158432, 99.7622910
25: -60.3617096, 39.9724846, -60.2863808, 39.8925629, -100.2542572, 100.2588654
26: -93.2930603, 51.4938927, -93.1175766, 51.4285583, -144.7215881, 144.6114655
27: -68.8240967, 44.5301514, -68.7248077, 44.4471817, -113.2712708, 113.2549591
28: -56.8558998, 36.7248840, -56.7707939, 36.6837959, -93.5396881, 93.4956818
29: -81.8784943, 54.7689476, -81.7835999, 54.7138405, -136.5923309, 136.5525513
30: -68.3340607, 37.4793243, -68.2261047, 37.4163132, -105.7503662, 105.7054291
31: -63.1773949, 30.9310150, -63.0266457, 30.8894844, -94.0668640, 93.9576569
32: -65.9895172, 48.4680557, -65.7670135, 48.4307594, -114.4202728, 114.2350616
33: -100.6081009, 58.8367386, -100.5060654, 58.6685829, -159.2766876, 159.3427734
34: -85.4974136, 44.8272095, -85.4300995, 44.7156601, -130.2130737, 130.2572937
35: -81.3222351, 47.6947556, -81.2432251, 47.5631714, -128.8854065, 128.9379730
36: -82.9230042, 48.6799126, -82.8324738, 48.6238861, -131.5468903, 131.5123901
37: -115.8563538, 48.3997116, -115.7309494, 48.3284302, -164.1847687, 164.1306610
38: -102.6691666, 63.9005890, -102.5427628, 63.8270569, -166.4962158, 166.4433441
39: -123.0440750, 55.0470276, -122.8940125, 54.9451981, -177.9892578, 177.9410248
40: -97.3141861, 47.7292519, -97.1998367, 47.6522141, -144.9663696, 144.9290771
41: -67.4594269, 40.3401299, -67.3295517, 40.2814598, -107.7408905, 107.6696777
42: -50.0471115, 45.4832535, -49.8635521, 45.4255371, -95.4726410, 95.3468018

Time for backsubstitution: 2.40 seconds

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
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1651
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
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1785
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
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
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
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
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
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1347
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
type: A, layer: 1, pos: 1551
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
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590838
time: 120.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4932411, upper bound: 52.4932407
time: 267.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 390.01 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3590838
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4265702, upper bound: 52.4467989
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590838
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 390.01
Output dim: 2, lower bound: -52.4932411, upper bound: 52.4932407

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.2446518, 45.7936478, -62.4421082, 45.8437271, -108.0883713, 108.2357559
1: -39.5786171, 41.6775360, -39.7258186, 41.7630310, -81.3416443, 81.4033508
2: -36.6101341, 43.6721802, -36.8183098, 43.7011986, -80.3113327, 80.4904861
3: -44.5591316, 51.7340050, -44.7839775, 51.8191986, -96.3783264, 96.5179825
4: -52.2391777, 40.4270668, -52.3752861, 40.4488258, -92.6879883, 92.8023529
5: -46.3619957, 56.7268677, -46.6106339, 56.8003044, -103.1622925, 103.3375015
6: -67.5634689, 41.2771606, -67.6900482, 41.4697227, -109.0331879, 108.9672089
7: -56.6375961, 52.6689224, -56.8982239, 52.8239594, -109.4615479, 109.5671463
8: -46.8457031, 46.9141502, -47.0602455, 46.9912567, -93.8369598, 93.9743958
9: -49.0585823, 52.1110954, -49.1873665, 52.3912811, -101.4498596, 101.2984619
10: -78.5686417, 75.7324219, -78.6530075, 76.1672211, -154.7358551, 154.3854370
11: -79.9553986, 52.4288940, -79.9614716, 52.6764793, -132.6318665, 132.3903656
12: -73.8274536, 57.6588326, -73.9409027, 58.3109398, -132.1383972, 131.5997314
13: -70.4974899, 65.9263763, -70.6579666, 66.2696152, -136.7670898, 136.5843353
14: -105.9806824, 56.2981339, -106.2201843, 56.8528061, -162.8334961, 162.5183105
15: -58.5138245, 50.2857704, -58.7172050, 50.4461594, -108.9599838, 109.0029755
16: -82.5367203, 65.8951874, -82.6353683, 66.0791168, -148.6158142, 148.5305481
17: -118.3614655, 77.6350021, -118.5333481, 78.2772217, -196.6386871, 196.1683502
18: -68.8076782, 41.8704605, -68.9560547, 41.9655457, -110.7732162, 110.8265152
19: -59.8336411, 24.8274117, -59.8995972, 24.8690338, -84.7026672, 84.7270050
20: -53.9260941, 32.1418915, -53.9916649, 32.2475433, -86.1736221, 86.1335526
21: -72.1421051, 36.3941879, -72.1801758, 36.5451965, -108.6873016, 108.5743637
22: -81.4473953, 47.6186485, -81.7036133, 47.8695869, -129.3169861, 129.3222656
23: -54.6212425, 34.5862274, -54.7371559, 34.6269684, -89.2482147, 89.3233795
24: -64.0368347, 34.5714798, -64.2640076, 34.6518555, -98.6886902, 98.8354874
25: -59.7910995, 39.3983307, -59.9367409, 39.5265427, -99.3176422, 99.3350677
26: -92.0764389, 49.8575630, -92.2771912, 50.3402328, -142.4166718, 142.1347504
27: -67.7789154, 44.1693497, -68.0538254, 44.2428780, -112.0217896, 112.2231750
28: -56.3283386, 36.4321098, -56.4573326, 36.4624519, -92.7907867, 92.8894424
29: -81.1510620, 53.6888313, -81.3329849, 54.0094986, -135.1605530, 135.0218201
30: -67.7303009, 36.6752472, -67.8907471, 36.8046150, -104.5349121, 104.5659943
31: -62.3263092, 30.5415039, -62.4746056, 30.5977592, -92.9240723, 93.0161133
32: -65.3142624, 47.6388168, -65.4129868, 47.8217278, -113.1359863, 113.0517883
33: -99.3053360, 58.2742691, -99.6021423, 58.2532425, -157.5585785, 157.8764038
34: -84.5876541, 44.3035736, -84.8445892, 44.3544846, -128.9421234, 129.1481628
35: -80.2128372, 47.2046738, -80.4632492, 47.2201996, -127.4330368, 127.6679230
36: -82.2277985, 48.2987823, -82.3652344, 48.3444519, -130.5722351, 130.6640015
37: -114.9188461, 47.9036102, -115.1515121, 47.9721222, -162.8909607, 163.0551147
38: -101.6383743, 63.3167229, -101.8951111, 63.4407883, -165.0791626, 165.2118378
39: -121.9574432, 54.5508194, -122.1701126, 54.6480522, -176.6054688, 176.7209320
40: -96.2241821, 47.2499847, -96.5173264, 47.3384819, -143.5626678, 143.7673035
41: -66.7496643, 39.6527786, -66.9395599, 39.7843208, -106.5339813, 106.5923309
42: -49.4823570, 44.2848396, -49.5569077, 44.4475937, -93.9299469, 93.8417435

Time for backsubstitution: 2.42 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 744
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
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
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1393
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
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3081350
time: 109.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3081350
time: 65.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -62.6196823, 46.0732956, -62.5018959, 45.9693069, -108.5889740, 108.5751953
1: -39.8316460, 41.8534241, -39.7607193, 41.8136024, -81.6452408, 81.6141434
2: -36.9545860, 43.9283752, -36.8508224, 43.8261414, -80.7807312, 80.7791977
3: -44.9485550, 52.0536537, -44.8223343, 51.9579201, -96.9064636, 96.8759842
4: -52.5088501, 40.5826988, -52.4081612, 40.5007820, -93.0096283, 92.9908600
5: -46.7817383, 57.0926552, -46.6562958, 56.9754944, -103.7572327, 103.7489471
6: -67.7994080, 41.5463295, -67.7413330, 41.5412178, -109.3406219, 109.2876587
7: -57.0832825, 52.9919243, -56.9586601, 52.9553757, -110.0386581, 109.9505768
8: -47.1729774, 47.1975517, -47.0852051, 47.1095581, -94.2825241, 94.2827606
9: -49.4019165, 52.5109749, -49.3313217, 52.4268036, -101.8287201, 101.8423004
10: -79.1332550, 76.4238586, -78.9097443, 76.2165222, -155.3497772, 155.3335876
11: -80.1318130, 52.7855148, -80.0500946, 52.7007027, -132.8325195, 132.8356018
12: -74.5409698, 58.5069046, -74.3056564, 58.3491058, -132.8900757, 132.8125610
13: -70.8417053, 66.3364639, -70.7931061, 66.3245239, -137.1662140, 137.1295776
14: -106.7635803, 56.9943008, -106.5671997, 56.8776817, -163.6412659, 163.5614929
15: -58.8299561, 50.5578766, -58.7879372, 50.5014496, -109.3314056, 109.3458099
16: -82.8038483, 66.2463303, -82.7303085, 66.1417236, -148.9455566, 148.9766388
17: -119.0077667, 78.3804092, -118.8388443, 78.2934265, -197.3011932, 197.2192535
18: -69.1378174, 42.1089821, -69.0581360, 42.0091972, -111.1469879, 111.1670990
19: -60.0456085, 24.9440556, -59.9703979, 24.8878326, -84.9334412, 84.9144516
20: -54.1526299, 32.3062744, -54.0687752, 32.2682419, -86.4208603, 86.3750458
21: -72.3931198, 36.6592522, -72.2700043, 36.5750732, -108.9681931, 108.9292526
22: -81.8905716, 48.0105400, -81.8830719, 47.9097977, -129.8003540, 129.8936157
23: -54.8608971, 34.7119827, -54.8035889, 34.6501694, -89.5110626, 89.5155716
24: -64.3146057, 34.7020798, -64.3122940, 34.6929970, -99.0075912, 99.0143738
25: -60.0028076, 39.6334953, -60.0024605, 39.5722427, -99.5750504, 99.6359558
26: -92.7724075, 50.5782013, -92.5964508, 50.3914757, -143.1638794, 143.1746521
27: -68.1366425, 44.3124733, -68.1020966, 44.2923431, -112.4289856, 112.4145660
28: -56.5605888, 36.5297775, -56.5140877, 36.4942017, -93.0547943, 93.0438614
29: -81.5146942, 54.1401405, -81.4799805, 54.0400085, -135.5547028, 135.6201172
30: -67.9922256, 36.9022827, -67.9468231, 36.8342857, -104.8265076, 104.8491058
31: -62.6149063, 30.6835861, -62.5415878, 30.6318665, -93.2467728, 93.2251740
32: -65.5144958, 47.8683205, -65.4697723, 47.8538818, -113.3683777, 113.3380814
33: -99.7259445, 58.4311676, -99.6555176, 58.3184280, -158.0443726, 158.0866852
34: -84.9667587, 44.5090981, -84.8953629, 44.4337845, -129.4005127, 129.4044647
35: -80.5869446, 47.3740349, -80.5047607, 47.2838058, -127.8707504, 127.8787994
36: -82.5031204, 48.4286423, -82.4230728, 48.3927765, -130.8959045, 130.8517151
37: -115.2609558, 48.0903664, -115.2227478, 48.0247154, -163.2856750, 163.3130951
38: -102.0607758, 63.5532227, -101.9517136, 63.5180321, -165.5787964, 165.5049286
39: -122.2783432, 54.7763863, -122.2226105, 54.7186852, -176.9970093, 176.9989929
40: -96.6292877, 47.5235329, -96.5642014, 47.4719009, -144.1011658, 144.0877380
41: -67.0307083, 39.8824234, -66.9817200, 39.8526573, -106.8833618, 106.8641357
42: -49.6506729, 44.5719719, -49.6093216, 44.4925232, -94.1431885, 94.1812897

Time for backsubstitution: 2.37 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
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
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
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
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1361
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
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3524131
time: 67.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4068221, upper bound: 52.3943030
time: 65.76 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5771446, 45.8676643, -63.0301476, 46.0672302, -108.6443787, 108.8977966
1: -39.7790108, 41.7398834, -40.0756607, 41.9016113, -81.6806107, 81.8155365
2: -36.9574966, 43.7188263, -37.4175758, 43.8809433, -80.8384323, 81.1363983
3: -44.9174576, 51.8082886, -45.4026566, 52.0475273, -96.9649811, 97.2109451
4: -52.6274300, 40.4955330, -53.0467529, 40.6356087, -93.2630310, 93.5422821
5: -46.7226791, 56.7931633, -47.2268944, 57.0495453, -103.7722244, 104.0200577
6: -67.6834183, 41.4686127, -67.9238815, 41.8175964, -109.5010071, 109.3924789
7: -56.9263535, 52.7340126, -57.4204254, 52.9845428, -109.9108963, 110.1544342
8: -47.1898232, 46.9805031, -47.6558952, 47.1829338, -94.3727570, 94.6363983
9: -49.1741333, 52.4023705, -49.4309273, 52.9031868, -102.0773163, 101.8332901
10: -78.7195892, 76.4019699, -79.1025620, 77.3124161, -156.0319977, 155.5045319
11: -80.0648041, 52.9234810, -80.2223892, 53.5091286, -133.5739288, 133.1458740
12: -73.9099121, 58.3753967, -74.3160706, 59.5419312, -133.4518433, 132.6914673
13: -70.6072693, 66.1067047, -70.8639221, 66.6207123, -137.2279816, 136.9706268
14: -106.1621323, 56.7396469, -106.6927872, 57.6179581, -163.7800903, 163.4324341
15: -58.8092346, 50.4032631, -59.2500801, 50.6791687, -109.4884033, 109.6533432
16: -82.6973572, 66.2498932, -82.9578705, 66.7094727, -149.4068298, 149.2077637
17: -118.4818268, 78.2586517, -118.8953400, 79.3485489, -197.8303680, 197.1539764
18: -68.9490585, 42.0994873, -69.2798843, 42.3669243, -111.3159790, 111.3793640
19: -59.9308891, 24.9869270, -60.1286583, 25.1385345, -85.0694275, 85.1155777
20: -54.0275154, 32.3034286, -54.2268600, 32.5271683, -86.5546875, 86.5302887
21: -72.2457733, 36.6593781, -72.4658127, 37.0037155, -109.2494812, 109.1251907
22: -81.5839310, 47.8468857, -81.9771729, 48.2758331, -129.8597717, 129.8240662
23: -54.7104454, 34.7473106, -54.9348717, 34.9088860, -89.6193237, 89.6821823
24: -64.1992569, 34.6328354, -64.5668869, 34.7537460, -98.9530029, 99.1997147
25: -59.8787270, 39.5362701, -60.1133194, 39.7765198, -99.6552353, 99.6495819
26: -92.1960678, 50.3322601, -92.6613007, 51.1677895, -143.3638458, 142.9935608
27: -68.0064850, 44.2274399, -68.4700470, 44.3541489, -112.3606339, 112.6974869
28: -56.4194984, 36.5085373, -56.6467438, 36.6068001, -93.0262909, 93.1552811
29: -81.2585602, 53.9929657, -81.5503464, 54.5293045, -135.7878723, 135.5433044
30: -67.8235931, 36.9194794, -68.0898819, 37.2307968, -105.0543900, 105.0093613
31: -62.4838905, 30.6606522, -62.8161011, 30.8045769, -93.2884674, 93.4767532
32: -65.4316788, 47.8603897, -65.6316910, 48.2048912, -113.6365662, 113.4920807
33: -99.6496735, 58.3822937, -100.2023621, 58.5279922, -158.1776733, 158.5846558
34: -84.8117828, 44.3984604, -85.2415771, 44.5718803, -129.3836670, 129.6400452
35: -80.5300293, 47.2916145, -81.0064850, 47.4390488, -127.9690781, 128.2980957
36: -82.4066925, 48.3780251, -82.6919098, 48.5110779, -130.9177704, 131.0699310
37: -115.1058807, 48.0258484, -115.4997787, 48.2074585, -163.3133392, 163.5256195
38: -101.8976288, 63.4216461, -102.3599014, 63.6539268, -165.5515442, 165.7815552
39: -122.2127762, 54.6297798, -122.6309814, 54.8231583, -177.0359192, 177.2607574
40: -96.4717407, 47.3016853, -96.9682312, 47.4762001, -143.9479370, 144.2699127
41: -66.8829956, 39.8099213, -67.1925507, 40.0702248, -106.9532089, 107.0024719
42: -49.5788879, 44.6676178, -49.7440338, 45.1119957, -94.6908875, 94.4116516

Time for backsubstitution: 2.44 seconds

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
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 682
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
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
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
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1743
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
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
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
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1703
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
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1690
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
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 821
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
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1429
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
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3810336
time: 82.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3540111
time: 79.99 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -62.9539337, 46.1469269, -63.0899849, 46.1920319, -109.1459656, 109.2369080
1: -40.0328522, 41.9159393, -40.1103935, 41.9531250, -81.9859772, 82.0263367
2: -37.3037720, 43.9747734, -37.4505348, 44.0024033, -81.3061752, 81.4253082
3: -45.3094940, 52.1275520, -45.4412994, 52.1859360, -97.4954300, 97.5688400
4: -52.8988495, 40.6515007, -53.0796661, 40.6878166, -93.5866699, 93.7311630
5: -47.1380081, 57.1589928, -47.2728348, 57.2173691, -104.3553772, 104.4318237
6: -67.9201050, 41.7421722, -67.9754944, 41.8917809, -109.8118896, 109.7176666
7: -57.3763084, 53.0562096, -57.4809418, 53.1170349, -110.4933472, 110.5371475
8: -47.5182037, 47.2637329, -47.6811829, 47.3010597, -94.8192596, 94.9449158
9: -49.5174637, 52.8035736, -49.5748978, 52.9387665, -102.4562225, 102.3784637
10: -79.2836151, 77.0958710, -79.3578262, 77.3611984, -156.6448059, 156.4537048
11: -80.2445984, 53.2831497, -80.3119431, 53.5345268, -133.7791138, 133.5950928
12: -74.6229935, 59.2267075, -74.6802826, 59.5808372, -134.2038116, 133.9069824
13: -70.9516449, 66.5187378, -70.9992218, 66.6766663, -137.6283112, 137.5179443
14: -106.9445190, 57.4368439, -107.0406342, 57.6426315, -164.5871582, 164.4774780
15: -59.1212540, 50.6767349, -59.3117523, 50.7347031, -109.8559570, 109.9884796
16: -82.9663925, 66.6067734, -83.0535278, 66.7712555, -149.7376404, 149.6603088
17: -119.1279526, 79.0077896, -119.2012329, 79.3659668, -198.4938965, 198.2090149
18: -69.2809601, 42.3377914, -69.3827667, 42.4106369, -111.6915970, 111.7205582
19: -60.1433907, 25.1025848, -60.1998444, 25.1576614, -85.3010559, 85.3024292
20: -54.2542572, 32.4690170, -54.3051491, 32.5479431, -86.8022003, 86.7741699
21: -72.4969025, 36.9257545, -72.5566864, 37.0339813, -109.5308838, 109.4824295
22: -82.0273514, 48.2390823, -82.1559067, 48.3158531, -130.3432007, 130.3949890
23: -54.9512863, 34.8738098, -55.0017967, 34.9319801, -89.8832550, 89.8756027
24: -64.4790192, 34.7639771, -64.6139069, 34.7948380, -99.2738571, 99.3778839
25: -60.0913086, 39.7724609, -60.1794739, 39.8219261, -99.9132385, 99.9519348
26: -92.8907471, 51.0570488, -92.9817657, 51.2186737, -144.1094208, 144.0388184
27: -68.3659897, 44.3708076, -68.5180969, 44.4028969, -112.7688751, 112.8889008
28: -56.6528435, 36.6061859, -56.7036514, 36.6383362, -93.2911835, 93.3098297
29: -81.6215897, 54.4458275, -81.6975174, 54.5594101, -136.1809998, 136.1433411
30: -68.0873718, 37.1476479, -68.1466141, 37.2608948, -105.3482666, 105.2942657
31: -62.7776756, 30.8034458, -62.8822899, 30.8387070, -93.6163788, 93.6857300
32: -65.6327972, 48.0918694, -65.6888885, 48.2375832, -113.8703766, 113.7807617
33: -100.0722656, 58.5393028, -100.2559891, 58.5945282, -158.6667938, 158.7952881
34: -85.1926117, 44.6037750, -85.2923889, 44.6517258, -129.8442993, 129.8961487
35: -80.9064865, 47.4603348, -81.0483246, 47.5036011, -128.4100952, 128.5086670
36: -82.6859741, 48.5100136, -82.7498474, 48.5600319, -131.2460022, 131.2598572
37: -115.4495773, 48.2133255, -115.5712509, 48.2607079, -163.7102814, 163.7845764
38: -102.3223190, 63.6591148, -102.4165115, 63.7305412, -166.0528564, 166.0756226
39: -122.5347595, 54.8562546, -122.6831818, 54.8943634, -177.4291229, 177.5394287
40: -96.8779602, 47.5757446, -97.0148773, 47.6104774, -144.4884338, 144.5906067
41: -67.1658707, 40.0335770, -67.2348938, 40.1295471, -107.2954025, 107.2684708
42: -49.7487259, 44.9597816, -49.7966766, 45.1570435, -94.9057617, 94.7564545

Time for backsubstitution: 2.38 seconds

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
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1625
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
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1703
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
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 821
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
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1429
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.4276363
time: 73.94 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3962307, upper bound: 52.4914514
time: 83.66 seconds

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

Time for backsubstitution: 2.39 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3081349
time: 64.61 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
time: 74.55 seconds

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

Time for backsubstitution: 2.38 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
time: 75.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524130
time: 122.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -62.9673843, 46.1008110, -63.1984138, 46.1128845, -109.0802689, 109.2992249
1: -39.9980240, 41.8522263, -40.1673279, 41.9362869, -81.9342957, 82.0195541
2: -37.3296585, 43.9578171, -37.6025543, 43.9074593, -81.2371216, 81.5603638
3: -45.2704926, 52.0266228, -45.5712967, 52.0952759, -97.3657684, 97.5979156
4: -53.0906830, 40.7847214, -53.2730942, 40.6751137, -93.7657776, 94.0578003
5: -47.0847168, 57.1023521, -47.4038391, 57.0967331, -104.1814423, 104.5061951
6: -67.9796600, 41.7898941, -67.9880981, 41.9637451, -109.9434052, 109.7779922
7: -57.2438278, 52.8617592, -57.5483742, 53.0245209, -110.2683487, 110.4101334
8: -47.6113434, 47.2500153, -47.8605042, 47.2224998, -94.8338318, 95.1105118
9: -49.4036064, 52.7779655, -49.4917068, 53.0753174, -102.4789124, 102.2696686
10: -79.2023926, 77.0867081, -79.1843262, 77.6537170, -156.8561096, 156.2710266
11: -80.5086823, 53.5289078, -80.2984772, 53.8232384, -134.3319244, 133.8273926
12: -74.5015030, 59.2468643, -74.3720093, 59.9980431, -134.4995422, 133.6188660
13: -70.8185883, 66.5532379, -70.9187927, 66.8092651, -137.6278534, 137.4720154
14: -106.6302948, 57.2723122, -106.7910461, 57.8984032, -164.5286865, 164.0633545
15: -59.2378654, 50.7399635, -59.4463882, 50.7484589, -109.9863205, 110.1863480
16: -83.0859528, 66.6587524, -83.0603485, 66.9069977, -149.9929504, 149.7191010
17: -119.0028076, 79.1153107, -118.9680481, 79.7995911, -198.8023987, 198.0833588
18: -69.2811279, 42.3132286, -69.3723068, 42.4510765, -111.7322083, 111.6855316
19: -60.1788597, 25.1402569, -60.1898079, 25.2055683, -85.3844299, 85.3300629
20: -54.2803955, 32.5063515, -54.2850342, 32.6245880, -86.9049835, 86.7913818
21: -72.5898743, 36.9394684, -72.5313568, 37.1420746, -109.7319489, 109.4708252
22: -81.8282166, 48.1299057, -82.0523911, 48.3768005, -130.2050171, 130.1822968
23: -54.9243088, 34.9156265, -54.9936523, 34.9792900, -89.9035873, 89.9092712
24: -64.5747910, 34.8206177, -64.7341309, 34.7862511, -99.3610382, 99.5547485
25: -60.0955048, 39.7218781, -60.1886406, 39.8374557, -99.9329605, 99.9105148
26: -92.5117416, 50.7558670, -92.7390900, 51.3625946, -143.8743134, 143.4949341
27: -68.4316864, 44.3642502, -68.6559448, 44.3852005, -112.8168869, 113.0201950
28: -56.6068459, 36.6132507, -56.7038155, 36.6433868, -93.2502289, 93.3170624
29: -81.4659195, 54.2999115, -81.6085205, 54.6722107, -136.1381226, 135.9084320
30: -68.0468445, 37.2252502, -68.1551361, 37.3700104, -105.4168549, 105.3803864
31: -62.8512764, 30.7701988, -62.9446220, 30.8435307, -93.6948013, 93.7148209
32: -65.7772675, 48.2111092, -65.6959839, 48.3822784, -114.1595459, 113.9070892
33: -100.1537170, 58.6611290, -100.4325867, 58.5880394, -158.7417603, 159.0937195
34: -85.0917282, 44.5977020, -85.3641129, 44.6200867, -129.7118225, 129.9618225
35: -80.9121170, 47.5094299, -81.1804352, 47.4868317, -128.3989258, 128.6898651
36: -82.6196365, 48.5352478, -82.7587128, 48.5654678, -131.1851044, 131.2939606
37: -115.4742584, 48.1954803, -115.6350555, 48.2638016, -163.7380371, 163.8305359
38: -102.2182617, 63.6387825, -102.4682007, 63.7357101, -165.9539795, 166.1069794
39: -122.6732101, 54.8035393, -122.8111267, 54.8619537, -177.5351562, 177.6146698
40: -96.8832932, 47.4289703, -97.1367569, 47.5017395, -144.3850403, 144.5657349
41: -67.1561279, 40.0643845, -67.2759247, 40.1870308, -107.3431549, 107.3403091
42: -49.8611031, 45.1452446, -49.7990875, 45.3531265, -95.2142334, 94.9443359

Time for backsubstitution: 2.41 seconds

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
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1789
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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 682
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
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1659
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
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
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
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 616
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
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1743
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1703
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
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1435
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
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 601
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
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 879
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
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 768
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
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3810336
time: 73.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3540111
time: 87.01 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.3405190, 46.3777580, -63.2580986, 46.2376556, -109.5781708, 109.6358490
1: -40.2502670, 42.0283356, -40.2019272, 41.9877663, -82.2380219, 82.2302628
2: -37.6743889, 44.2127495, -37.6353035, 44.0288010, -81.7031860, 81.8480530
3: -45.6544876, 52.3406410, -45.6100006, 52.2331696, -97.8876495, 97.9506378
4: -53.3596077, 40.9395142, -53.3051529, 40.7274170, -94.0870209, 94.2446671
5: -47.4938011, 57.4629364, -47.4496574, 57.2645874, -104.7583618, 104.9125824
6: -68.2119904, 42.0573235, -68.0396576, 42.0374680, -110.2494583, 110.0969849
7: -57.6879272, 53.1852303, -57.6086655, 53.1576271, -110.8455505, 110.7938995
8: -47.9371262, 47.5315170, -47.8849945, 47.3404846, -95.2776108, 95.4164963
9: -49.7453728, 53.1765785, -49.6356201, 53.1103897, -102.8557587, 102.8121948
10: -79.7635651, 77.7766800, -79.4393082, 77.7017746, -157.4653320, 157.2159882
11: -80.6647034, 53.8871269, -80.3883362, 53.8484192, -134.5131226, 134.2754669
12: -75.2113037, 60.0949173, -74.7359314, 60.0362968, -135.2476044, 134.8308411
13: -71.1609650, 66.9611359, -71.0540466, 66.8640747, -138.0250244, 138.0151825
14: -107.4103241, 57.9675179, -107.1386261, 57.9224548, -165.3327789, 165.1061401
15: -59.5093842, 51.0088921, -59.4775581, 50.8043633, -110.3137512, 110.4864502
16: -83.3428497, 67.0200958, -83.1562042, 66.9669189, -150.3097687, 150.1763000
17: -119.6471863, 79.8590088, -119.2736053, 79.8155212, -199.4626923, 199.1325989
18: -69.6100464, 42.5514297, -69.4744110, 42.4950104, -112.1050568, 112.0258331
19: -60.3905144, 25.2542324, -60.2609673, 25.2247524, -85.6152649, 85.5151978
20: -54.5046043, 32.6713943, -54.3636055, 32.6453476, -87.1499481, 87.0350037
21: -72.8392029, 37.2051620, -72.6223526, 37.1722832, -110.0114746, 109.8274994
22: -82.2785568, 48.5178642, -82.2304535, 48.4172935, -130.6958466, 130.7483215
23: -55.1641922, 35.0411453, -55.0607185, 35.0022430, -90.1664352, 90.1018677
24: -64.8521729, 34.9507294, -64.7804260, 34.8274460, -99.6796188, 99.7311554
25: -60.3085136, 39.9566574, -60.2545891, 39.8829422, -100.1914520, 100.2112427
26: -93.2013931, 51.4693985, -93.0581436, 51.4136848, -144.6150665, 144.5275421
27: -68.7874908, 44.5091591, -68.7028656, 44.4336929, -113.2211838, 113.2120209
28: -56.8396339, 36.7099075, -56.7607498, 36.6747932, -93.5144272, 93.4706573
29: -81.8329391, 54.7501144, -81.7551804, 54.7025604, -136.5354919, 136.5052795
30: -68.3104248, 37.4517059, -68.2119293, 37.3997803, -105.7101974, 105.6636353
31: -63.1504211, 30.9119892, -63.0102539, 30.8778725, -94.0282898, 93.9222412
32: -65.9669647, 48.4406509, -65.7533417, 48.4145851, -114.3815308, 114.1939926
33: -100.5744171, 58.8142586, -100.4859695, 58.6548080, -159.2292175, 159.3002167
34: -85.4720306, 44.8017693, -85.4147720, 44.6998978, -130.1719055, 130.2165222
35: -81.2870941, 47.6757393, -81.2221985, 47.5516014, -128.8386993, 128.8979340
36: -82.8970947, 48.6646767, -82.8167572, 48.6145935, -131.5116882, 131.4814301
37: -115.8145142, 48.3815994, -115.7061920, 48.3172760, -164.1317749, 164.0877991
38: -102.6388397, 63.8760033, -102.5246124, 63.8121109, -166.4509430, 166.4006042
39: -122.9894562, 55.0280647, -122.8618240, 54.9334946, -177.9229431, 177.8898926
40: -97.2862015, 47.7034531, -97.1830444, 47.6361351, -144.9223328, 144.8865051
41: -67.4410095, 40.2783775, -67.3184357, 40.2434235, -107.6844330, 107.5968094
42: -50.0277748, 45.4357452, -49.8518867, 45.3974686, -95.4252319, 95.2876282

Time for backsubstitution: 2.40 seconds

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
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
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
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1703
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
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1435
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
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1429
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
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.4276363
time: 84.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3943030
time: 190.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 277.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3081350
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3081350
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3524131
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4068221, upper bound: 52.3943030
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3810336
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3540111
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.4276363
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.3962307, upper bound: 52.4914514
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3081349
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524130
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3810336
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3540111
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.4276363
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 277.27
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3943030

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -61.9721375, 45.7179108, -61.9960899, 45.7195740, -107.6917038, 107.7139969
1: -39.4236450, 41.6115646, -39.4747086, 41.6543350, -81.0779800, 81.0862656
2: -36.3562050, 43.6195107, -36.4041443, 43.6152573, -79.9714661, 80.0236511
3: -44.2653084, 51.6583138, -44.3020096, 51.6940308, -95.9593353, 95.9603271
4: -51.9349442, 40.3600235, -51.8837204, 40.3392296, -92.2741699, 92.2437439
5: -46.0960159, 56.6569099, -46.1746216, 56.6854820, -102.7814941, 102.8315277
6: -67.4467621, 41.1383324, -67.4992294, 41.2460709, -108.6928329, 108.6375580
7: -56.4358139, 52.6058388, -56.5659256, 52.7204971, -109.1563110, 109.1717529
8: -46.6243095, 46.8381004, -46.7019997, 46.8668251, -93.4911346, 93.5401001
9: -48.9011765, 51.9532089, -48.9314117, 52.1392403, -101.0404205, 100.8846130
10: -78.4301682, 75.1718369, -78.4276733, 75.2567902, -153.6869507, 153.5995178
11: -79.8432388, 51.9168243, -79.7757263, 51.8468819, -131.6901093, 131.6925354
12: -73.7413940, 57.1523361, -73.8000183, 57.4796524, -131.2210388, 130.9523621
13: -70.2446899, 65.7718658, -70.2479553, 66.0165176, -136.2612000, 136.0198059
14: -105.8011475, 55.8875618, -105.9258041, 56.1764908, -161.9776306, 161.8133698
15: -58.2525673, 50.1718864, -58.2958565, 50.2587128, -108.5112762, 108.4677429
16: -82.3624725, 65.6203003, -82.3488541, 65.6474075, -148.0098877, 147.9691467
17: -118.2298508, 77.0073471, -118.3176727, 77.2438583, -195.4737091, 195.3250122
18: -68.6619034, 41.5880203, -68.7151031, 41.5013046, -110.1632080, 110.3031235
19: -59.7374268, 24.6095963, -59.7421799, 24.5134583, -84.2508850, 84.3517761
20: -53.8293304, 31.9652290, -53.8320961, 31.9567966, -85.7861252, 85.7973175
21: -72.0339508, 36.0789948, -72.0041809, 36.0263939, -108.0603333, 108.0831680
22: -81.3249817, 47.3802452, -81.5062866, 47.4778175, -128.8027954, 128.8865356
23: -54.5361977, 34.3653259, -54.5970726, 34.2662582, -88.8024521, 88.9624023
24: -63.9277191, 34.4584961, -64.0871429, 34.4627113, -98.3904266, 98.5456390
25: -59.6999626, 39.2327271, -59.7879601, 39.2554550, -98.9554138, 99.0206909
26: -91.9607010, 49.4556427, -92.0865860, 49.6739349, -141.6346283, 141.5422363
27: -67.6123962, 44.0404053, -67.7845001, 44.0345917, -111.6469803, 111.8249054
28: -56.2402077, 36.3200073, -56.3131638, 36.2811852, -92.5213852, 92.6331635
29: -81.0539932, 53.3629761, -81.1760254, 53.4736404, -134.5276337, 134.5390015
30: -67.6317291, 36.4294815, -67.7282181, 36.4038887, -104.0356064, 104.1576920
31: -62.1837044, 30.3325481, -62.2365646, 30.2560844, -92.4397736, 92.5691147
32: -65.1847458, 47.4994812, -65.2008057, 47.5986290, -112.7833710, 112.7002716
33: -98.9344635, 58.1798935, -98.9944611, 58.0982857, -157.0327454, 157.1743469
34: -84.3415451, 44.2132034, -84.4382782, 44.2083588, -128.5498810, 128.6514587
35: -79.8687134, 47.1248703, -79.8972626, 47.0917435, -126.9604492, 127.0221329
36: -81.9680099, 48.2274551, -81.9409332, 48.2263908, -130.1943970, 130.1683960
37: -114.7182312, 47.7910919, -114.8263321, 47.7900620, -162.5082855, 162.6174316
38: -101.3841400, 63.2179108, -101.4822311, 63.2800064, -164.6641235, 164.7001343
39: -121.6232758, 54.4781036, -121.6227570, 54.5297470, -176.1530151, 176.1008606
40: -95.9900360, 47.2008858, -96.1332092, 47.2570457, -143.2470703, 143.3340912
41: -66.6231842, 39.5217628, -66.7323990, 39.5737991, -106.1969757, 106.2541656
42: -49.3891106, 43.9796257, -49.4034538, 43.9520264, -93.3411331, 93.3830795

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
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
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 681
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1638
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
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1777
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
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 617
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
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
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 948
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
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1341
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
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 821
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
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 1359
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
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 680
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
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
time: 80.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 63.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -62.2124367, 45.7827873, -62.4917679, 46.0370636, -108.2494965, 108.2745514
1: -39.5590973, 41.6671219, -39.7458916, 41.8211517, -81.3802490, 81.4130096
2: -36.5851936, 43.6643906, -36.8204956, 43.8557396, -80.4409332, 80.4848862
3: -44.5272217, 51.7212334, -44.7769814, 52.0566216, -96.5838470, 96.4982071
4: -52.2056580, 40.4155731, -52.3821907, 40.5488319, -92.7544861, 92.7977600
5: -46.3332138, 56.7152710, -46.6122971, 57.0200920, -103.3533020, 103.3275681
6: -67.5445557, 41.2221375, -67.7272339, 41.4606819, -109.0052338, 108.9493713
7: -56.6124611, 52.6551476, -56.9235191, 52.8711319, -109.4835968, 109.5786591
8: -46.8195305, 46.9021645, -47.0686340, 47.0811653, -93.9006805, 93.9707947
9: -49.0261688, 52.0842133, -49.2100830, 52.4149055, -101.4410706, 101.2942963
10: -78.5512085, 75.6707306, -78.9867172, 76.1541595, -154.7053680, 154.6574402
11: -79.9394073, 52.3885307, -80.2980957, 52.6460724, -132.5854797, 132.6866150
12: -73.8120575, 57.6113663, -74.3076401, 58.2980614, -132.1101074, 131.9190063
13: -70.4573975, 65.9047699, -70.6714783, 66.4812851, -136.9386902, 136.5762482
14: -105.9544601, 56.2564850, -106.5486221, 56.8291130, -162.7835693, 162.8051147
15: -58.4534302, 50.2691727, -58.7278900, 50.4796944, -108.9331207, 108.9970627
16: -82.5130615, 65.8447113, -82.7271118, 66.0814056, -148.5944519, 148.5718231
17: -118.3438034, 77.5737152, -118.9291763, 78.2390366, -196.5828247, 196.5028839
18: -68.7859421, 41.8397789, -69.2535248, 41.9667130, -110.7526550, 111.0932999
19: -59.8215866, 24.8036442, -60.1074181, 24.8586159, -84.6801910, 84.9110565
20: -53.9120331, 32.1248474, -54.1843681, 32.2542877, -86.1663208, 86.3092117
21: -72.1256409, 36.3634415, -72.4753876, 36.5346832, -108.6603088, 108.8388290
22: -81.4214096, 47.5906258, -81.7759705, 47.8743896, -129.2958069, 129.3665924
23: -54.6096535, 34.5622978, -54.9373627, 34.6298637, -89.2395096, 89.4996567
24: -64.0176468, 34.5560532, -64.3798523, 34.6584244, -98.6760712, 98.9359055
25: -59.7775078, 39.3762131, -60.0392342, 39.5292931, -99.3068008, 99.4154510
26: -92.0555649, 49.8172951, -92.6662750, 50.3472366, -142.4028015, 142.4835663
27: -67.7530441, 44.1459084, -68.1108246, 44.2495422, -112.0025864, 112.2567291
28: -56.3173370, 36.4158096, -56.5867271, 36.4862213, -92.8035507, 93.0025330
29: -81.1303024, 53.6517754, -81.4238129, 53.9910011, -135.1212921, 135.0755920
30: -67.7156219, 36.6513367, -68.1198883, 36.8262978, -104.5419083, 104.7712097
31: -62.3092690, 30.5176849, -62.6964302, 30.5837746, -92.8930435, 93.2141113
32: -65.2930603, 47.6164970, -65.4454346, 47.8402100, -113.1332703, 113.0619354
33: -99.2713776, 58.2616043, -99.6166840, 58.5383682, -157.8097534, 157.8782959
34: -84.5578766, 44.2903976, -84.8578491, 44.4752350, -129.0331116, 129.1482544
35: -80.1747437, 47.1950302, -80.4591217, 47.4382820, -127.6130219, 127.6541519
36: -82.1943817, 48.2888260, -82.3619308, 48.4282761, -130.6226501, 130.6507568
37: -114.8820572, 47.8892708, -115.2028427, 48.0227127, -162.9047699, 163.0921021
38: -101.5959473, 63.3038483, -101.9208679, 63.5394821, -165.1354065, 165.2247162
39: -121.9119263, 54.5405579, -122.1877899, 54.8654938, -176.7774200, 176.7283325
40: -96.1913376, 47.2410583, -96.5590363, 47.4635620, -143.6549072, 143.8000946
41: -66.7303925, 39.6164513, -66.9758148, 39.8127022, -106.5430908, 106.5922699
42: -49.4678078, 44.2438164, -49.6014023, 44.4644012, -93.9322052, 93.8452148

Time for backsubstitution: 2.43 seconds

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
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 681
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 617
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1674
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 825
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
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1352
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
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
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
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
time: 71.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
time: 78.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.3466835, 45.9977112, -62.0557709, 45.8452415, -108.1919098, 108.0534821
1: -39.6763077, 41.7873917, -39.5095749, 41.7043457, -81.3806534, 81.2969589
2: -36.7002106, 43.8759499, -36.4365082, 43.7404099, -80.4406204, 80.3124542
3: -44.6535187, 51.9782867, -44.3405762, 51.8338203, -96.4873276, 96.3188629
4: -52.2042847, 40.5152016, -51.9165344, 40.3907204, -92.5949936, 92.4317322
5: -46.5149879, 57.0229187, -46.2202301, 56.8606224, -103.3756104, 103.2431488
6: -67.6820831, 41.4087563, -67.5499344, 41.3189011, -109.0009613, 108.9586868
7: -56.8796616, 52.9295425, -56.6274757, 52.8517838, -109.7314453, 109.5570221
8: -46.9512100, 47.1217117, -46.7268867, 46.9851837, -93.9363937, 93.8486023
9: -49.2448349, 52.3523788, -49.0756874, 52.1747208, -101.4195480, 101.4280624
10: -78.9959259, 75.8627472, -78.6847992, 75.3060303, -154.3019409, 154.5475311
11: -80.0177765, 52.2738953, -79.8641663, 51.8705025, -131.8882751, 132.1380615
12: -74.4552612, 57.9999695, -74.1649246, 57.5176506, -131.9729156, 132.1648865
13: -70.5898590, 66.1809616, -70.3836670, 66.0702820, -136.6601410, 136.5646362
14: -106.5848083, 56.5833130, -106.2728424, 56.2015228, -162.7863312, 162.8561554
15: -58.5701675, 50.4432373, -58.3684044, 50.3133888, -108.8835449, 108.8116455
16: -82.6284714, 65.9693604, -82.4430542, 65.7108154, -148.3392639, 148.4124146
17: -118.8763504, 77.7523956, -118.6233673, 77.2594757, -196.1358337, 196.3757629
18: -68.9907990, 41.8261490, -68.8165283, 41.5450249, -110.5358276, 110.6426697
19: -59.9493828, 24.7261658, -59.8127556, 24.5320625, -84.4814453, 84.5389252
20: -54.0562248, 32.1292725, -53.9084129, 31.9774284, -86.0336533, 86.0376816
21: -72.2852554, 36.3437500, -72.0931473, 36.0562172, -108.3414612, 108.4368973
22: -81.7685547, 47.7708817, -81.6862793, 47.5180206, -129.2865753, 129.4571533
23: -54.7751808, 34.4909172, -54.6627312, 34.2894363, -89.0646057, 89.1536331
24: -64.2044678, 34.5884705, -64.1345444, 34.5036926, -98.7081528, 98.7230072
25: -59.9114685, 39.4674606, -59.8532257, 39.3012505, -99.2127228, 99.3206711
26: -92.6573105, 50.1741333, -92.4079819, 49.7254868, -142.3827972, 142.5820923
27: -67.9692001, 44.1837692, -67.8324280, 44.0846863, -112.0538788, 112.0161972
28: -56.4717827, 36.4175491, -56.3693390, 36.3126831, -92.7844696, 92.7868881
29: -81.4183273, 53.8135262, -81.3236771, 53.5041122, -134.9224396, 135.1372070
30: -67.8930588, 36.6562881, -67.7837753, 36.4334831, -104.3265228, 104.4400635
31: -62.4690514, 30.4745922, -62.3029289, 30.2899609, -92.7590103, 92.7775116
32: -65.3844299, 47.7283783, -65.2570648, 47.6303596, -113.0147781, 112.9854431
33: -99.3548203, 58.3371544, -99.0477066, 58.1626129, -157.5174255, 157.3848572
34: -84.7199097, 44.4191284, -84.4888916, 44.2870941, -129.0070038, 128.9080200
35: -80.2424927, 47.2950592, -79.9386292, 47.1543655, -127.3968582, 127.2336884
36: -82.2415771, 48.3559189, -81.9990540, 48.2739906, -130.5155487, 130.3549805
37: -115.0597000, 47.9775925, -114.8971939, 47.8421173, -162.9018250, 162.8747864
38: -101.8053360, 63.4540863, -101.5384216, 63.3565903, -165.1619263, 164.9925079
39: -121.9440308, 54.7033234, -121.6749496, 54.5997238, -176.5437469, 176.3782654
40: -96.3947906, 47.4742393, -96.1801453, 47.3904114, -143.7852020, 143.6543884
41: -66.9031982, 39.7536316, -66.7743225, 39.6458664, -106.5490494, 106.5279541
42: -49.5559921, 44.2651787, -49.4552574, 43.9978294, -93.5538177, 93.7204361

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 649
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
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1400
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
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1431
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
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 651
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
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
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 768
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
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 680
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 69.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 63.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.5873756, 46.0624580, -62.5512962, 46.1626511, -108.7500305, 108.6137314
1: -39.8120995, 41.8429756, -39.7805786, 41.8715363, -81.6836166, 81.6235504
2: -36.9296150, 43.9205399, -36.8526764, 43.9807587, -80.9103699, 80.7732162
3: -44.9165421, 52.0408211, -44.8152275, 52.1958923, -97.1124344, 96.8560410
4: -52.4752960, 40.5711288, -52.4146805, 40.6006203, -93.0759125, 92.9858093
5: -46.7528915, 57.0809784, -46.6576157, 57.1952248, -103.9481201, 103.7385941
6: -67.7804260, 41.4916573, -67.7780228, 41.5320435, -109.3124695, 109.2696838
7: -57.0579147, 52.9783821, -56.9842949, 53.0026398, -110.0605545, 109.9626770
8: -47.1468163, 47.1855087, -47.0933228, 47.1995430, -94.3463593, 94.2788239
9: -49.3695221, 52.4840965, -49.3536491, 52.4504051, -101.8199310, 101.8377380
10: -79.1158600, 76.3621140, -79.2438202, 76.2031708, -155.3190308, 155.6059265
11: -80.1154938, 52.7451477, -80.3865051, 52.6699142, -132.7854004, 133.1316528
12: -74.5254822, 58.4594193, -74.6725082, 58.3359070, -132.8613739, 133.1319275
13: -70.8013229, 66.3146820, -70.8061981, 66.5364838, -137.3377991, 137.1208801
14: -106.7371750, 56.9525452, -106.8961792, 56.8538742, -163.5910187, 163.8487244
15: -58.7697983, 50.5411606, -58.7986069, 50.5347176, -109.3045197, 109.3397675
16: -82.7797318, 66.1963882, -82.8217621, 66.1436768, -148.9234009, 149.0181580
17: -118.9898834, 78.3191681, -119.2349548, 78.2548676, -197.2447357, 197.5541229
18: -69.1157684, 42.0782204, -69.3562469, 42.0101051, -111.1258698, 111.4344635
19: -60.0335236, 24.9202995, -60.1783562, 24.8773193, -84.9108429, 85.0986481
20: -54.1385689, 32.2891273, -54.2616882, 32.2747765, -86.4133453, 86.5508118
21: -72.3766327, 36.6284943, -72.5648804, 36.5644913, -108.9411240, 109.1933746
22: -81.8649063, 47.9823914, -81.9560242, 47.9143562, -129.7792664, 129.9384155
23: -54.8491936, 34.6880417, -55.0037308, 34.6528854, -89.5020752, 89.6917725
24: -64.2953796, 34.6865730, -64.4288177, 34.6992874, -98.9946594, 99.1153870
25: -59.9890976, 39.6113281, -60.1048470, 39.5747604, -99.5638504, 99.7161713
26: -92.7512817, 50.5376740, -92.9862900, 50.3985519, -143.1498413, 143.5239563
27: -68.1107025, 44.2892838, -68.1596146, 44.2987900, -112.4094849, 112.4488907
28: -56.5494690, 36.5134354, -56.6433830, 36.5175591, -93.0670166, 93.1568146
29: -81.4938202, 54.1031036, -81.5713196, 54.0211906, -135.5150146, 135.6744232
30: -67.9774933, 36.8782272, -68.1758499, 36.8557472, -104.8332367, 105.0540771
31: -62.5972176, 30.6597672, -62.7640381, 30.6176205, -93.2148361, 93.4238052
32: -65.4932632, 47.8460693, -65.5017700, 47.8720894, -113.3653564, 113.3478394
33: -99.6919250, 58.4184875, -99.6698303, 58.6031456, -158.2950745, 158.0883179
34: -84.9368896, 44.4958878, -84.9084625, 44.5544777, -129.4913635, 129.4043579
35: -80.5488434, 47.3644714, -80.5004120, 47.5017281, -128.0505676, 127.8648834
36: -82.4695587, 48.4184685, -82.4196014, 48.4768219, -130.9463806, 130.8380737
37: -115.2242432, 48.0760803, -115.2735519, 48.0747757, -163.2990112, 163.3496399
38: -102.0185776, 63.5402298, -101.9770966, 63.6163025, -165.6348877, 165.5173340
39: -122.2328720, 54.7662277, -122.2398376, 54.9360962, -177.1689453, 177.0060577
40: -96.5963974, 47.5143318, -96.6056061, 47.5975723, -144.1939697, 144.1199341
41: -67.0113754, 39.8463898, -67.0177460, 39.8826790, -106.8940582, 106.8641357
42: -49.6360512, 44.5309334, -49.6537666, 44.5096664, -94.1457214, 94.1846924

Time for backsubstitution: 2.41 seconds

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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1654
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
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1573
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
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1431
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
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1615
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
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
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
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 80.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 74.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.3007736, 45.7914047, -62.5718040, 45.9417267, -108.2425003, 108.3631973
1: -39.6212769, 41.6735649, -39.8146057, 41.7914429, -81.4127197, 81.4881744
2: -36.6998825, 43.6657104, -36.9909515, 43.7935333, -80.4934082, 80.6566620
3: -44.6197090, 51.7322578, -44.9083290, 51.9220428, -96.5417328, 96.6405792
4: -52.3182716, 40.4278297, -52.5370102, 40.5239143, -92.8421783, 92.9648285
5: -46.4538574, 56.7227402, -46.7803802, 56.9331245, -103.3869705, 103.5031204
6: -67.5663147, 41.3246880, -67.7298813, 41.5799332, -109.1462250, 109.0545502
7: -56.7214584, 52.6706123, -57.0769882, 52.8799057, -109.6013641, 109.7476044
8: -46.9645615, 46.9040337, -47.2838364, 47.0568352, -94.0213928, 94.1878662
9: -49.0151291, 52.2410355, -49.1693077, 52.6371040, -101.6522293, 101.4103394
10: -78.5803375, 75.8341980, -78.8735123, 76.3747101, -154.9550171, 154.7077026
11: -79.9521942, 52.4061966, -80.0336151, 52.6613808, -132.6135712, 132.4397888
12: -73.8232574, 57.8631401, -74.1732483, 58.6915703, -132.5148315, 132.0363770
13: -70.3530807, 65.9523468, -70.4473114, 66.3638153, -136.7168884, 136.3996582
14: -105.9819260, 56.3260765, -106.3961639, 56.9326973, -162.9146118, 162.7222443
15: -58.5385513, 50.2894325, -58.8045616, 50.4896088, -109.0281601, 109.0939789
16: -82.5228271, 65.9666290, -82.6683044, 66.2394714, -148.7622986, 148.6349335
17: -118.3491821, 77.6248932, -118.6769104, 78.2958832, -196.6450653, 196.3018036
18: -68.8011017, 41.8143539, -69.0338898, 41.8921356, -110.6932220, 110.8482437
19: -59.8340569, 24.7669010, -59.9684563, 24.7747612, -84.6088181, 84.7353592
20: -53.9302139, 32.1249084, -54.0665703, 32.2306099, -86.1608276, 86.1914825
21: -72.1369476, 36.3418236, -72.2872314, 36.4765778, -108.6135254, 108.6290436
22: -81.4606934, 47.6043854, -81.7742462, 47.8745155, -129.3352051, 129.3786316
23: -54.6250954, 34.5241776, -54.7929382, 34.5395622, -89.1646576, 89.3171158
24: -64.0885849, 34.5187378, -64.3815308, 34.5642624, -98.6528473, 98.9002609
25: -59.7871323, 39.3684540, -59.9619942, 39.4981155, -99.2852325, 99.3304443
26: -92.0795441, 49.9261322, -92.4690399, 50.4899101, -142.5694427, 142.3951721
27: -67.8375244, 44.0970306, -68.1896744, 44.1418037, -111.9793243, 112.2866974
28: -56.3309937, 36.3948822, -56.4995842, 36.4200172, -92.7510071, 92.8944626
29: -81.1604309, 53.6633148, -81.3884811, 53.9815140, -135.1419373, 135.0517883
30: -67.7246780, 36.6706543, -67.9248199, 36.8202438, -104.5449219, 104.5954742
31: -62.3392067, 30.4500237, -62.5701180, 30.4560814, -92.7952881, 93.0201340
32: -65.3012390, 47.7185287, -65.4154587, 47.9712524, -113.2724915, 113.1339874
33: -99.2760162, 58.2873611, -99.5863571, 58.3722420, -157.6482391, 157.8737183
34: -84.5635986, 44.3079948, -84.8285980, 44.4237671, -128.9873657, 129.1365967
35: -80.1827316, 47.2118683, -80.4301529, 47.3094597, -127.4921875, 127.6420135
36: -82.1430283, 48.3067017, -82.2543640, 48.3908539, -130.5338745, 130.5610657
37: -114.9023590, 47.9132004, -115.1640091, 48.0218849, -162.9242401, 163.0772095
38: -101.6372757, 63.3213501, -101.9291534, 63.4875183, -165.1247864, 165.2504883
39: -121.8754120, 54.5573044, -122.0722122, 54.7030258, -176.5784302, 176.6295166
40: -96.2348633, 47.2523422, -96.5750656, 47.3945198, -143.6293793, 143.8274078
41: -66.7548065, 39.6737404, -66.9795761, 39.8486633, -106.6034698, 106.6533203
42: -49.4853859, 44.3564644, -49.5881119, 44.5969772, -94.0823669, 93.9445801

Time for backsubstitution: 2.43 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 682
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
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 681
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1668
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
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1297
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
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1449
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
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 988
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
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
time: 72.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3741897
time: 81.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -62.5462265, 45.8571281, -63.0869141, 46.2642822, -108.8104782, 108.9440308
1: -39.7603874, 41.7296600, -40.1006508, 41.9612198, -81.7216034, 81.8303070
2: -36.9339676, 43.7112122, -37.4258080, 44.0373611, -80.9713287, 81.1370239
3: -44.8868027, 51.7959099, -45.4015808, 52.2881126, -97.1749039, 97.1974945
4: -52.5955544, 40.4844284, -53.0612907, 40.7403717, -93.3359222, 93.5457153
5: -46.6951218, 56.7819366, -47.2332344, 57.2720413, -103.9671631, 104.0151672
6: -67.6651459, 41.4170952, -67.9712830, 41.8169479, -109.4820862, 109.3883820
7: -56.9022369, 52.7197533, -57.4527664, 53.0335388, -109.9357758, 110.1725082
8: -47.1650391, 46.9687347, -47.6707420, 47.2754555, -94.4404755, 94.6394730
9: -49.1423340, 52.3761444, -49.4567986, 52.9317703, -102.0740967, 101.8329315
10: -78.7026825, 76.3430023, -79.4393921, 77.3112488, -156.0139313, 155.7823944
11: -80.0496292, 52.8853722, -80.5913086, 53.4921455, -133.5417786, 133.4766846
12: -73.8949280, 58.3305473, -74.6872559, 59.5383263, -133.4332581, 133.0177917
13: -70.5750122, 66.0862045, -70.8941498, 66.8383636, -137.4133759, 136.9803467
14: -106.1369629, 56.6994247, -107.0279922, 57.5990486, -163.7360077, 163.7274170
15: -58.7601128, 50.3873787, -59.2919960, 50.7270622, -109.4871750, 109.6793747
16: -82.6744843, 66.2009583, -83.0582581, 66.7219849, -149.3964691, 149.2592163
17: -118.4649887, 78.1999893, -119.2968369, 79.3209076, -197.7858887, 197.4968109
18: -68.9280930, 42.0694275, -69.5809174, 42.3720016, -111.3000870, 111.6503220
19: -59.9193115, 24.9637985, -60.3393860, 25.1317749, -85.0510788, 85.3031769
20: -54.0139427, 32.2870483, -54.4288139, 32.5372543, -86.5511932, 86.7158585
21: -72.2298889, 36.6296196, -72.7658997, 36.9971619, -109.2270355, 109.3955154
22: -81.5569305, 47.8194695, -82.0530319, 48.2879333, -129.8448639, 129.8724976
23: -54.6992645, 34.7241249, -55.1414833, 34.9152679, -89.6145325, 89.8656082
24: -64.1808319, 34.6176453, -64.6755753, 34.7624817, -98.9433136, 99.2932129
25: -59.8656578, 39.5146675, -60.2185287, 39.7845459, -99.6502075, 99.7332001
26: -92.1757202, 50.2933350, -93.0551147, 51.1807594, -143.3564453, 143.3484497
27: -67.9816895, 44.2051239, -68.5274048, 44.3634186, -112.3451080, 112.7325287
28: -56.4089546, 36.4934616, -56.7784081, 36.6352806, -93.0442276, 93.2718658
29: -81.2384415, 53.9569969, -81.6467056, 54.5159607, -135.7543945, 135.6036987
30: -67.8093872, 36.8966484, -68.3290405, 37.2572556, -105.0666351, 105.2256927
31: -62.4673462, 30.6372375, -63.0371284, 30.7941055, -93.2614441, 93.6743622
32: -65.4110184, 47.8386765, -65.6727448, 48.2279472, -113.6389618, 113.5114136
33: -99.6169662, 58.3702469, -100.2212067, 58.8190727, -158.4360352, 158.5914612
34: -84.7826920, 44.3858185, -85.2584381, 44.7020874, -129.4847717, 129.6442566
35: -80.4930573, 47.2823792, -81.0085449, 47.6719322, -128.1649933, 128.2909241
36: -82.3738480, 48.3686905, -82.6936493, 48.5977745, -130.9715881, 131.0623474
37: -115.0698700, 48.0123901, -115.5567169, 48.2654572, -163.3353271, 163.5691071
38: -101.8574982, 63.4091988, -102.3945923, 63.7538986, -165.6113892, 165.8037872
39: -122.1686325, 54.6200676, -122.6579514, 55.0497208, -177.2183533, 177.2780151
40: -96.4398117, 47.2931137, -97.0155640, 47.6035805, -144.0433960, 144.3086853
41: -66.8642883, 39.7731285, -67.2347412, 40.1011047, -106.9653931, 107.0078659
42: -49.5648537, 44.6296387, -49.7955780, 45.1438141, -94.7086487, 94.4252167

Time for backsubstitution: 2.41 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 682
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
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 681
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
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
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
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
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
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
time: 67.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4381855
time: 73.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.6771927, 46.0710754, -62.6316261, 46.0667076, -108.7438965, 108.7026978
1: -39.8747787, 41.8494759, -39.8493423, 41.8432541, -81.7180328, 81.6988144
2: -37.0457458, 43.9219551, -37.0238266, 43.9151917, -80.9609375, 80.9457855
3: -45.0108337, 52.0518303, -44.9469452, 52.0607796, -97.0716019, 96.9987717
4: -52.5893593, 40.5836182, -52.5699921, 40.5757713, -93.1651154, 93.1536026
5: -46.8681641, 57.0887032, -46.8262520, 57.1003761, -103.9685211, 103.9149551
6: -67.8024139, 41.5994682, -67.7809601, 41.6557350, -109.4581451, 109.3804321
7: -57.1693153, 52.9935188, -57.1375313, 53.0129356, -110.1822357, 110.1310349
8: -47.2926826, 47.1875305, -47.3090515, 47.1751709, -94.4678497, 94.4965820
9: -49.3588066, 52.6415787, -49.3135910, 52.6728287, -102.0316315, 101.9551697
10: -79.1452484, 76.5276413, -79.1293335, 76.4234238, -155.5686646, 155.6569824
11: -80.1297302, 52.7663460, -80.1225281, 52.6861801, -132.8159180, 132.8888702
12: -74.5367126, 58.7140121, -74.5376434, 58.7303810, -133.2670898, 133.2516479
13: -70.6984177, 66.3633881, -70.5835495, 66.4187546, -137.1171722, 136.9469299
14: -106.7653046, 57.0229607, -106.7440033, 56.9573250, -163.7226257, 163.7669678
15: -58.8517914, 50.5621223, -58.8707390, 50.5450897, -109.3968735, 109.4328613
16: -82.7907715, 66.3201294, -82.7632904, 66.3028412, -149.0936127, 149.0834198
17: -118.9958191, 78.3737946, -118.9829788, 78.3128815, -197.3086853, 197.3567505
18: -69.1320496, 42.0517540, -69.1359558, 41.9358368, -111.0678864, 111.1877060
19: -60.0463943, 24.8825378, -60.0392838, 24.7937202, -84.8401184, 84.9218216
20: -54.1571999, 32.2902298, -54.1443748, 32.2513733, -86.4085693, 86.4346008
21: -72.3884125, 36.6079178, -72.3772659, 36.5066986, -108.8951035, 108.9851761
22: -81.9046402, 47.9964600, -81.9533463, 47.9144440, -129.8190765, 129.9497986
23: -54.8651085, 34.6505280, -54.8592987, 34.5626411, -89.4277496, 89.5098267
24: -64.3672791, 34.6493225, -64.4283600, 34.6052856, -98.9725647, 99.0776825
25: -59.9993973, 39.6042137, -60.0277214, 39.5436249, -99.5430145, 99.6319351
26: -92.7750549, 50.6491928, -92.7901535, 50.5410538, -143.3160858, 143.4393463
27: -68.1964645, 44.2407188, -68.2371216, 44.1914177, -112.3878784, 112.4778442
28: -56.5637093, 36.4924965, -56.5560417, 36.4515839, -93.0152893, 93.0485382
29: -81.5242386, 54.1154861, -81.5365524, 54.0116119, -135.5358582, 135.6520386
30: -67.9874725, 36.8986778, -67.9815063, 36.8501892, -104.8376312, 104.8801880
31: -62.6296310, 30.5928078, -62.6363335, 30.4900150, -93.1196365, 93.2291412
32: -65.5019073, 47.9495964, -65.4721527, 48.0036392, -113.5055466, 113.4217529
33: -99.6984558, 58.4446335, -99.6398468, 58.4382019, -158.1366577, 158.0844727
34: -84.9437943, 44.5137444, -84.8792801, 44.5029449, -129.4467316, 129.3930206
35: -80.5589905, 47.3813934, -80.4719238, 47.3730927, -127.9320831, 127.8533173
36: -82.4215698, 48.4372482, -82.3122025, 48.4390411, -130.8605957, 130.7494507
37: -115.2451019, 48.1004715, -115.2349625, 48.0746613, -163.3197632, 163.3354340
38: -102.0608673, 63.5585670, -101.9854507, 63.5639038, -165.6247711, 165.5440216
39: -122.1973724, 54.7833481, -122.1246033, 54.7736244, -176.9709930, 176.9079437
40: -96.6408081, 47.5263138, -96.6218033, 47.5287552, -144.1695557, 144.1481171
41: -67.0366669, 39.8998146, -67.0217896, 39.9121628, -106.9488297, 106.9216003
42: -49.6540680, 44.6472092, -49.6402092, 44.6418381, -94.2958984, 94.2874146

Time for backsubstitution: 2.40 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
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
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1668
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
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
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
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
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
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 680
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
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
time: 76.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
time: 78.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.9229393, 46.1364288, -63.1465111, 46.3892212, -109.3121490, 109.2829285
1: -40.0142136, 41.9056740, -40.1351547, 42.0130386, -82.0272522, 82.0408325
2: -37.2802086, 43.9671555, -37.4584808, 44.1589241, -81.4391327, 81.4256287
3: -45.2787781, 52.1151505, -45.4399872, 52.4268188, -97.7055969, 97.5551224
4: -52.8670006, 40.6403618, -53.0940285, 40.7924728, -93.6594696, 93.7343826
5: -47.1102791, 57.1476746, -47.2788391, 57.4395370, -104.5498047, 104.4265137
6: -67.9017944, 41.6908493, -68.0224457, 41.8911438, -109.7929306, 109.7132950
7: -57.3519897, 53.0421410, -57.5129280, 53.1663284, -110.5183182, 110.5550690
8: -47.4934692, 47.2519379, -47.6958237, 47.3937874, -94.8872528, 94.9477615
9: -49.4856949, 52.7773476, -49.6004868, 52.9674416, -102.4531326, 102.3778305
10: -79.2667236, 77.0368423, -79.6950684, 77.3597641, -156.6264954, 156.7319031
11: -80.2291107, 53.2450294, -80.6808624, 53.5174026, -133.7464905, 133.9258881
12: -74.6079254, 59.1819000, -75.0515747, 59.5770264, -134.1849518, 134.2334747
13: -70.9190063, 66.4981003, -71.0292206, 66.8947449, -137.8137512, 137.5273132
14: -106.9192123, 57.3965340, -107.3763123, 57.6236000, -164.5428009, 164.7728424
15: -59.0727463, 50.6607361, -59.3536453, 50.7828178, -109.8555527, 110.0143814
16: -82.9431458, 66.5584259, -83.1537094, 66.7831268, -149.7262573, 149.7121277
17: -119.1109695, 78.9491425, -119.6030502, 79.3380203, -198.4489441, 198.5521851
18: -69.2598114, 42.3076363, -69.6843567, 42.4155121, -111.6753235, 111.9919739
19: -60.1317101, 25.0794754, -60.4105682, 25.1508236, -85.2825241, 85.4900360
20: -54.2406921, 32.4525757, -54.5069427, 32.5578766, -86.7985687, 86.9595184
21: -72.4809875, 36.8959656, -72.8564301, 37.0273056, -109.5082855, 109.7523880
22: -82.0006943, 48.2117233, -82.2322464, 48.3276482, -130.3283386, 130.4439697
23: -54.9399986, 34.8506165, -55.2084084, 34.9382477, -89.8782349, 90.0590210
24: -64.4605331, 34.7487564, -64.7230988, 34.8033409, -99.2638702, 99.4718475
25: -60.0781326, 39.7508240, -60.2845955, 39.8297577, -99.9078903, 100.0354156
26: -92.8702469, 51.0179749, -93.3758698, 51.2317200, -144.1019592, 144.3938446
27: -68.3412018, 44.3487396, -68.5763855, 44.4121590, -112.7533569, 112.9251251
28: -56.6422043, 36.5910797, -56.8353806, 36.6664581, -93.3086624, 93.4264526
29: -81.6014099, 54.4098969, -81.7944031, 54.5458221, -136.1472321, 136.2042999
30: -68.0731049, 37.1246490, -68.3857956, 37.2871094, -105.3602066, 105.5104370
31: -62.7605019, 30.7800407, -63.1042709, 30.8280067, -93.5885086, 93.8843079
32: -65.6121674, 48.0703201, -65.7294617, 48.2606201, -113.8727875, 113.7997818
33: -100.0395050, 58.5272446, -100.2747345, 58.8852768, -158.9247589, 158.8019714
34: -85.1634979, 44.5911484, -85.3092575, 44.7818565, -129.9453583, 129.9004059
35: -80.8695374, 47.4512100, -81.0502548, 47.7363358, -128.6058655, 128.5014648
36: -82.6530457, 48.5004425, -82.7513123, 48.6470490, -131.3000946, 131.2517395
37: -115.4135590, 48.1998711, -115.6277008, 48.3183250, -163.7318726, 163.8275757
38: -102.2824402, 63.6465759, -102.4508057, 63.8304138, -166.1128540, 166.0973816
39: -122.4906464, 54.8464851, -122.7101059, 55.1209259, -177.6115723, 177.5565796
40: -96.8460007, 47.5668983, -97.0619888, 47.7384338, -144.5844421, 144.6288910
41: -67.1471100, 39.9970398, -67.2768250, 40.1620178, -107.3091278, 107.2738647
42: -49.7346611, 44.9214668, -49.8482933, 45.1881866, -94.9228516, 94.7697601

Time for backsubstitution: 2.46 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1599
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
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
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
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 99.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4893945
time: 77.18 seconds

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

Time for backsubstitution: 2.43 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
time: 65.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 75.89 seconds

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

Time for backsubstitution: 2.43 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
time: 79.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
time: 73.92 seconds

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

Time for backsubstitution: 2.40 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 72.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 73.71 seconds

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

Time for backsubstitution: 2.44 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
time: 74.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 166.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.6910744, 46.0251503, -62.7398605, 45.9877014, -108.6787720, 108.7650146
1: -39.8405838, 41.7859421, -39.9065628, 41.8262634, -81.6668472, 81.6925049
2: -37.0723763, 43.9050140, -37.1760178, 43.8202438, -80.8926239, 81.0810318
3: -44.9729042, 51.9511452, -45.0769920, 51.9701996, -96.9431000, 97.0281372
4: -52.7819633, 40.7172852, -52.7633972, 40.5635986, -93.3455658, 93.4806824
5: -46.8162651, 57.0322266, -46.9572754, 56.9806366, -103.7969055, 103.9895020
6: -67.8630753, 41.6424332, -67.7946014, 41.7251434, -109.5882187, 109.4370270
7: -57.0388756, 52.7982559, -57.2045555, 52.9202576, -109.9591370, 110.0028076
8: -47.3863602, 47.1739426, -47.4884529, 47.0966377, -94.4829941, 94.6623917
9: -49.2453423, 52.6166687, -49.2303162, 52.8092384, -102.0545807, 101.8469849
10: -79.0645599, 76.5195770, -78.9559937, 76.7165985, -155.7811584, 155.4755707
11: -80.3984070, 53.0120964, -80.1104736, 52.9755859, -133.3739929, 133.1225586
12: -74.4153442, 58.7348480, -74.2294235, 59.1476593, -133.5629883, 132.9642639
13: -70.5645447, 66.3979492, -70.5020905, 66.5527496, -137.1172943, 136.9000397
14: -106.4508591, 56.8581123, -106.4951477, 57.2121201, -163.6629639, 163.3532562
15: -58.9602661, 50.6265869, -58.9922714, 50.5592651, -109.5195312, 109.6188507
16: -82.9124756, 66.3693695, -82.7721939, 66.4304962, -149.3429565, 149.1415558
17: -118.8707809, 78.4811096, -118.7501373, 78.7463684, -197.6171570, 197.2312469
18: -69.1323853, 42.0274506, -69.1267929, 41.9752617, -111.1076508, 111.1542435
19: -60.0816650, 24.9206772, -60.0303841, 24.8420906, -84.9237518, 84.9510651
20: -54.1835785, 32.3278275, -54.1250534, 32.3279877, -86.5115662, 86.4528656
21: -72.4816971, 36.6222763, -72.3533630, 36.6151352, -109.0968170, 108.9756317
22: -81.7046204, 47.8879700, -81.8500366, 47.9758682, -129.6804810, 129.7380066
23: -54.8392067, 34.6928902, -54.8522224, 34.6101761, -89.4493713, 89.5451126
24: -64.4636002, 34.7069397, -64.5485077, 34.5967178, -99.0603180, 99.2554474
25: -60.0034904, 39.5544319, -60.0375175, 39.5591812, -99.5626678, 99.5919495
26: -92.3951340, 50.3500023, -92.5470810, 50.6849213, -143.0800476, 142.8970795
27: -68.2623596, 44.2342339, -68.3757629, 44.1729202, -112.4352722, 112.6099930
28: -56.5181541, 36.4996986, -56.5568924, 36.4563675, -92.9745178, 93.0565948
29: -81.3675003, 53.9706573, -81.4467621, 54.1247330, -135.4922180, 135.4174194
30: -67.9480133, 36.9763451, -67.9903107, 36.9594154, -104.9074249, 104.9666595
31: -62.7035942, 30.5598049, -62.6980133, 30.4951839, -93.1987762, 93.2578201
32: -65.6481705, 48.0693970, -65.4802017, 48.1489716, -113.7971344, 113.5495911
33: -99.7799225, 58.5667610, -99.8163986, 58.4326096, -158.2125244, 158.3831482
34: -84.8431549, 44.5076294, -84.9507446, 44.4725075, -129.3156586, 129.4583740
35: -80.5646896, 47.4304352, -80.6039581, 47.3579788, -127.9226685, 128.0343933
36: -82.3558578, 48.4635925, -82.3212585, 48.4458847, -130.8017426, 130.7848511
37: -115.2694321, 48.0831718, -115.2987366, 48.0788918, -163.3483276, 163.3819122
38: -101.9575958, 63.5379028, -102.0373535, 63.5689430, -165.5265350, 165.5752563
39: -122.3350067, 54.7311134, -122.2519608, 54.7423096, -177.0773010, 176.9830627
40: -96.6459351, 47.3793411, -96.7427979, 47.4203262, -144.0662537, 144.1221313
41: -67.0280762, 39.9267998, -67.0630722, 39.9641876, -106.9922638, 106.9898682
42: -49.7689247, 44.8333931, -49.6437607, 44.8382683, -94.6071930, 94.4771576

Time for backsubstitution: 2.43 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 682
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 994
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
time: 62.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4772059, upper bound: 52.3741897
time: 88.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -62.9364433, 46.0903397, -63.2551003, 46.3098450, -109.2462692, 109.3454437
1: -39.9794235, 41.8419342, -40.1924858, 41.9957809, -81.9752045, 82.0344162
2: -37.3060570, 43.9502602, -37.6109810, 44.0638313, -81.3698883, 81.5612411
3: -45.2397919, 52.0143127, -45.5702324, 52.3356628, -97.5754547, 97.5845490
4: -53.0588493, 40.7736511, -53.2878151, 40.7798157, -93.8386688, 94.0614624
5: -47.0571671, 57.0911827, -47.4101486, 57.3192062, -104.3763733, 104.5013275
6: -67.9614868, 41.7378311, -68.0362015, 41.9629936, -109.9244843, 109.7740326
7: -57.2197189, 52.8471794, -57.5806847, 53.0734596, -110.2931824, 110.4278564
8: -47.5866013, 47.2382660, -47.8753967, 47.3149834, -94.9015732, 95.1136475
9: -49.3717842, 52.7517357, -49.5178833, 53.1038208, -102.4756012, 102.2696152
10: -79.1856461, 77.0277557, -79.5209122, 77.6529922, -156.8386383, 156.5486755
11: -80.4938660, 53.4908218, -80.6667023, 53.8066635, -134.3005219, 134.1575317
12: -74.4866028, 59.2020416, -74.7430420, 59.9945068, -134.4811096, 133.9450836
13: -70.7866058, 66.5325775, -70.9495239, 67.0254593, -137.8120575, 137.4820862
14: -106.6051559, 57.2320099, -107.1260986, 57.8788490, -164.4840088, 164.3581085
15: -59.1888924, 50.7241669, -59.4854698, 50.7967300, -109.9856262, 110.2096405
16: -83.0632553, 66.6096420, -83.1605606, 66.9174118, -149.9806671, 149.7702026
17: -118.9860458, 79.0566025, -119.3694305, 79.7719116, -198.7579651, 198.4260101
18: -69.2600250, 42.2830582, -69.6722794, 42.4560814, -111.7161102, 111.9553375
19: -60.1672096, 25.1171741, -60.4005661, 25.1989441, -85.3661499, 85.5177307
20: -54.2669144, 32.4899521, -54.4869766, 32.6347275, -86.9016418, 86.9769287
21: -72.5740967, 36.9097214, -72.8313599, 37.1356125, -109.7097092, 109.7410812
22: -81.8006592, 48.1026421, -82.1280289, 48.3898544, -130.1905212, 130.2306671
23: -54.9131851, 34.8924751, -55.2000656, 34.9857750, -89.8989563, 90.0925369
24: -64.5563354, 34.8055038, -64.8419876, 34.7951126, -99.3514481, 99.6474915
25: -60.0824432, 39.7003212, -60.2936440, 39.8459282, -99.9283752, 99.9939499
26: -92.4914017, 50.7169075, -93.1326981, 51.3759079, -143.8673096, 143.8496094
27: -68.4068604, 44.3421249, -68.7120361, 44.3946342, -112.8014832, 113.0541611
28: -56.5963097, 36.5983543, -56.8352356, 36.6717911, -93.2680969, 93.4335785
29: -81.4457703, 54.2639809, -81.7048492, 54.6594009, -136.1051636, 135.9688263
30: -68.0326538, 37.2023277, -68.3940811, 37.3965912, -105.4292450, 105.5964050
31: -62.8342590, 30.7467861, -63.1640511, 30.8332367, -93.6674957, 93.9108353
32: -65.7567444, 48.1894302, -65.7377014, 48.4053001, -114.1620331, 113.9271088
33: -100.1209564, 58.6491394, -100.4515686, 58.8789978, -158.9999542, 159.1007080
34: -85.0625534, 44.5851364, -85.3808594, 44.7502251, -129.8127594, 129.9660034
35: -80.8751526, 47.5002365, -81.1825485, 47.7194138, -128.5945740, 128.6827850
36: -82.5867157, 48.5258675, -82.7606583, 48.6511116, -131.2378235, 131.2865295
37: -115.4381180, 48.1820068, -115.6920090, 48.3223534, -163.7604675, 163.8740234
38: -102.1781921, 63.6262703, -102.5032654, 63.8351326, -166.0133057, 166.1295319
39: -122.6291199, 54.7939072, -122.8383331, 55.0882225, -177.7173462, 177.6322327
40: -96.8513184, 47.4203796, -97.1839447, 47.6290550, -144.4803467, 144.6043243
41: -67.1375427, 40.0272789, -67.3185883, 40.2170334, -107.3545761, 107.3458633
42: -49.8471909, 45.1072388, -49.8507195, 45.3854752, -95.2326660, 94.9579620

Time for backsubstitution: 2.50 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 682
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1435
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
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1341
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
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
time: 75.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4381855
time: 76.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -63.0638924, 46.3025551, -62.7995796, 46.1126862, -109.1765671, 109.1021347
1: -40.0925446, 41.9618835, -39.9411736, 41.8780441, -81.9705811, 81.9030609
2: -37.4167252, 44.1602669, -37.2087936, 43.9418106, -81.3585358, 81.3690643
3: -45.3560486, 52.2654190, -45.1157875, 52.1084518, -97.4645004, 97.3811951
4: -53.0505791, 40.8719444, -52.7957230, 40.6155777, -93.6661530, 93.6676636
5: -47.2243195, 57.3930397, -47.0032043, 57.1479301, -104.3722534, 104.3962402
6: -68.0948944, 41.9103394, -67.8458023, 41.8002930, -109.8951874, 109.7561340
7: -57.4805412, 53.1223526, -57.2652893, 53.0538330, -110.5343552, 110.3876343
8: -47.7118225, 47.4556961, -47.5129395, 47.2147827, -94.9266052, 94.9686356
9: -49.5873756, 53.0146675, -49.3745155, 52.8443336, -102.4317017, 102.3891830
10: -79.6264954, 77.2091675, -79.2114487, 76.7645721, -156.3910675, 156.4206238
11: -80.5515137, 53.3708382, -80.1996460, 53.0003319, -133.5518494, 133.5704803
12: -75.1255264, 59.5825043, -74.5935669, 59.1859436, -134.3114624, 134.1760712
13: -70.9078293, 66.8049927, -70.6383209, 66.6066132, -137.5144348, 137.4432983
14: -107.2318344, 57.5531006, -106.8427963, 57.2361946, -164.4680328, 164.3959045
15: -59.2342415, 50.8948784, -59.0288048, 50.6148071, -109.8490448, 109.9236832
16: -83.1682434, 66.7285461, -82.8675537, 66.4920044, -149.6602478, 149.5960999
17: -119.5156479, 79.2246399, -119.0558548, 78.7620163, -198.2776642, 198.2804871
18: -69.4603271, 42.2646255, -69.2279358, 42.0191116, -111.4794159, 111.4925613
19: -60.2932739, 25.0346947, -60.1010818, 24.8611603, -85.1544342, 85.1357727
20: -54.4081764, 32.4926529, -54.2030754, 32.3487778, -86.7569427, 86.6957245
21: -72.7314148, 36.8877106, -72.4435272, 36.6453018, -109.3767014, 109.3312378
22: -82.1559525, 48.2760468, -82.0285187, 48.0163002, -130.1722565, 130.3045654
23: -55.0782890, 34.8182678, -54.9186134, 34.6331787, -89.7114334, 89.7368774
24: -64.7399368, 34.8364410, -64.5943756, 34.6378975, -99.3778381, 99.4308167
25: -60.2165413, 39.7889023, -60.1032181, 39.6047134, -99.8212585, 99.8921204
26: -93.0853729, 51.0614967, -92.8668518, 50.7363434, -143.8217163, 143.9283447
27: -68.6176758, 44.3795357, -68.4219513, 44.2223282, -112.8400040, 112.8014832
28: -56.7503662, 36.5962410, -56.6134796, 36.4879150, -93.2382660, 93.2097168
29: -81.7351532, 54.4201469, -81.5941772, 54.1549911, -135.8901367, 136.0143127
30: -68.2103424, 37.2026672, -68.0468140, 36.9891510, -105.1994934, 105.2494659
31: -63.0006485, 30.7015305, -62.7631187, 30.5294075, -93.5300446, 93.4646454
32: -65.8375549, 48.2984619, -65.5370178, 48.1810341, -114.0185852, 113.8354797
33: -100.2004242, 58.7201347, -99.8696136, 58.4987869, -158.6992188, 158.5897522
34: -85.2228088, 44.7120667, -85.0013123, 44.5516777, -129.7744904, 129.7133789
35: -80.9394531, 47.5976219, -80.6456451, 47.4217682, -128.3612061, 128.2432556
36: -82.6326904, 48.5918427, -82.3791199, 48.4942703, -131.1269531, 130.9709625
37: -115.6096344, 48.2690887, -115.3696823, 48.1319656, -163.7416077, 163.6387634
38: -102.3771210, 63.7748260, -102.0934753, 63.6448975, -166.0219879, 165.8682861
39: -122.6509933, 54.9552917, -122.3021240, 54.8132858, -177.4642639, 177.2574158
40: -97.0486145, 47.6537590, -96.7891541, 47.5547180, -144.6033325, 144.4429169
41: -67.3118896, 40.1411896, -67.1053925, 40.0227928, -107.3346863, 107.2465744
42: -49.9341469, 45.1231880, -49.6959877, 44.8823853, -94.8165283, 94.8191681

Time for backsubstitution: 2.47 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1654
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
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1396
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
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 601
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
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 673
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
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
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1560
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
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
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
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
time: 75.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
time: 70.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -63.3095474, 46.3673172, -63.3146400, 46.4347458, -109.7442856, 109.6819458
1: -40.2316437, 42.0180435, -40.2269096, 42.0476837, -82.2793274, 82.2449493
2: -37.6508026, 44.2051888, -37.6435318, 44.1852951, -81.8360977, 81.8487244
3: -45.6237144, 52.3283043, -45.6087799, 52.4739151, -98.0976257, 97.9370880
4: -53.3277893, 40.9283981, -53.3198814, 40.8320274, -94.1598206, 94.2482758
5: -47.4660492, 57.4516869, -47.4557648, 57.4867363, -104.9527893, 104.9074478
6: -68.1937561, 42.0056305, -68.0875397, 42.0366821, -110.2304382, 110.0931702
7: -57.6635628, 53.1708565, -57.6410828, 53.2068481, -110.8704071, 110.8119354
8: -47.9123917, 47.5197258, -47.8998222, 47.4331512, -95.3455429, 95.4195480
9: -49.7134628, 53.1503868, -49.6615639, 53.1389389, -102.8523941, 102.8119507
10: -79.7468491, 77.7177429, -79.7763062, 77.7007294, -157.4475708, 157.4940491
11: -80.6494751, 53.8490143, -80.7565002, 53.8317719, -134.4812469, 134.6054993
12: -75.1962814, 60.0500679, -75.1071243, 60.0325737, -135.2288513, 135.1571960
13: -71.1285553, 66.9404144, -71.0845795, 67.0808258, -138.2093811, 138.0249939
14: -107.3850327, 57.9271851, -107.4742432, 57.9027634, -165.2877960, 165.4014282
15: -59.4611320, 50.9929695, -59.5164719, 50.8523941, -110.3135223, 110.5094452
16: -83.3197708, 66.9716949, -83.2563629, 66.9767303, -150.2964630, 150.2280579
17: -119.6302414, 79.8003540, -119.6753311, 79.7876282, -199.4178619, 199.4756775
18: -69.5888138, 42.5211449, -69.7751389, 42.4998779, -112.0886841, 112.2962799
19: -60.3788338, 25.2311668, -60.4716568, 25.2180920, -85.5969086, 85.7028198
20: -54.4910889, 32.6549377, -54.5653152, 32.6553764, -87.1464691, 87.2202377
21: -72.8233948, 37.1753922, -72.9220123, 37.1657906, -109.9891815, 110.0974045
22: -82.2513351, 48.4906616, -82.3065491, 48.4299774, -130.6813049, 130.7972107
23: -55.1529808, 35.0179825, -55.2671661, 35.0086670, -90.1616516, 90.2851410
24: -64.8336639, 34.9355240, -64.8886490, 34.8361549, -99.6698151, 99.8241730
25: -60.2953453, 39.9351120, -60.3595619, 39.8912392, -100.1865845, 100.2946625
26: -93.1808472, 51.4301910, -93.4520493, 51.4271049, -144.6079254, 144.8822327
27: -68.7626266, 44.4872971, -68.7595825, 44.4430618, -113.2056885, 113.2468796
28: -56.8290291, 36.6949234, -56.8922806, 36.7029762, -93.5320053, 93.5871964
29: -81.8127060, 54.7142334, -81.8519592, 54.6894951, -136.5021973, 136.5661926
30: -68.2961502, 37.4286575, -68.4507446, 37.4261780, -105.7223282, 105.8794022
31: -63.1330833, 30.8885384, -63.2305145, 30.8673954, -94.0004807, 94.1190491
32: -65.9464111, 48.4191360, -65.7945251, 48.4375229, -114.3839340, 114.2136612
33: -100.5416107, 58.8022652, -100.5047607, 58.9453888, -159.4869995, 159.3070221
34: -85.4428177, 44.7891617, -85.4315262, 44.8299942, -130.2728119, 130.2206879
35: -81.2500610, 47.6667213, -81.2242050, 47.7840309, -129.0340881, 128.8909302
36: -82.8641052, 48.6551361, -82.8184128, 48.7006226, -131.5647278, 131.4735413
37: -115.7784348, 48.3680954, -115.7629929, 48.3755417, -164.1539764, 164.1310883
38: -102.5989914, 63.8634491, -102.5593033, 63.9115829, -166.5105438, 166.4227295
39: -122.9453201, 55.0183640, -122.8884277, 55.1598434, -178.1051636, 177.9067993
40: -97.2541809, 47.6946373, -97.2300034, 47.7639847, -145.0181274, 144.9246368
41: -67.4223175, 40.2414398, -67.3608170, 40.2743111, -107.6966248, 107.6022568
42: -50.0137978, 45.3973846, -49.9035187, 45.4291649, -95.4429474, 95.3009033

Time for backsubstitution: 2.46 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1651
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
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 985
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
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1704
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
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
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
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1347
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
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
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 936
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
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4342985, upper bound: 52.4891413
time: 71.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4893945
time: 101.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 175.13 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3741897
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4381855
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4893945
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.4772059, upper bound: 52.3741897
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4381855
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.4342985, upper bound: 52.4891413
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 175.13
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4893945

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -61.4155846, 45.4030151, -61.7239609, 45.6319656, -107.0475464, 107.1269608
1: -39.1092072, 41.4404335, -39.3233414, 41.5805664, -80.6897736, 80.7637711
2: -35.8377533, 43.3666229, -36.1320648, 43.5620651, -79.3998184, 79.4986877
3: -43.8382492, 51.4036674, -44.0737610, 51.5933266, -95.4315720, 95.4774323
4: -51.2204170, 40.0619125, -51.4982910, 40.2549515, -91.4753723, 91.5601883
5: -45.6702423, 56.3938065, -45.9558334, 56.5955963, -102.2658310, 102.3496399
6: -67.0685196, 40.6098480, -67.3641357, 40.9721870, -108.0407104, 107.9739685
7: -56.0470657, 52.3504868, -56.3949356, 52.5957985, -108.6428604, 108.7454224
8: -46.0191498, 46.4988976, -46.3820877, 46.7852592, -92.8044128, 92.8809738
9: -48.5208206, 51.3973694, -48.8246498, 51.8395767, -100.3603973, 100.2220154
10: -77.6469421, 73.9598999, -78.2725906, 74.5741959, -152.2211304, 152.2324829
11: -79.2394714, 50.9113388, -79.6423798, 51.2793655, -130.5188141, 130.5537109
12: -73.0984650, 55.8933563, -73.6919708, 56.7762756, -129.8747253, 129.5853271
13: -69.9309082, 65.1032486, -70.1432037, 65.6895294, -135.6204376, 135.2464447
14: -105.1629257, 55.0348015, -105.7510986, 55.6936493, -160.8565674, 160.7858887
15: -57.5472946, 49.7531929, -57.9451523, 50.1315804, -107.6788635, 107.6983337
16: -81.7906647, 64.8383636, -82.1747284, 65.2155533, -147.0061951, 147.0130920
17: -117.6440811, 75.6550827, -118.1937943, 76.4868622, -194.1309509, 193.8488617
18: -68.1176376, 41.3008194, -68.4925232, 41.3717613, -109.4893951, 109.7933350
19: -59.3932571, 24.4302998, -59.6130486, 24.4317036, -83.8249512, 84.0433502
20: -53.4966698, 31.6887741, -53.7248230, 31.8078156, -85.3044891, 85.4135971
21: -71.6054688, 35.7156563, -71.8792877, 35.8287125, -107.4341736, 107.5949402
22: -80.7878265, 47.0009651, -81.2509460, 47.3108368, -128.0986481, 128.2519073
23: -54.2716217, 34.0955009, -54.4942284, 34.1308098, -88.4024124, 88.5897293
24: -63.3120117, 34.2298660, -63.7647972, 34.3968239, -97.7088318, 97.9946594
25: -59.3565636, 38.9789429, -59.6263161, 39.1507950, -98.5073547, 98.6052551
26: -91.5452805, 48.9129181, -91.9265747, 49.3759003, -140.9211578, 140.8394928
27: -66.8879395, 43.7712479, -67.4076996, 43.9662056, -110.8541260, 111.1789398
28: -55.9531670, 36.1194229, -56.1774635, 36.2088013, -92.1619644, 92.2968903
29: -80.6513748, 53.0071182, -80.9974670, 53.2835846, -133.9349518, 134.0045776
30: -67.3104858, 35.9901733, -67.6031647, 36.1797256, -103.4901886, 103.5933380
31: -61.6894836, 30.1714268, -62.0243874, 30.1818085, -91.8712921, 92.1958160
32: -64.8110046, 47.0496330, -65.0661163, 47.3536987, -112.1646957, 112.1157532
33: -98.1363678, 57.7118492, -98.5597916, 57.9910889, -156.1274414, 156.2716370
34: -83.8603516, 43.8671799, -84.1795425, 44.1101875, -127.9705353, 128.0467224
35: -79.1829987, 46.6931763, -79.5218048, 47.0006714, -126.1836548, 126.2149811
36: -81.5209045, 47.9825363, -81.7088394, 48.1467056, -129.6676025, 129.6913757
37: -114.0403061, 47.4992294, -114.4826279, 47.6895065, -161.7297974, 161.9818573
38: -100.8051224, 62.9354401, -101.1894302, 63.1697388, -163.9748535, 164.1248627
39: -120.8505554, 54.2545280, -121.2229996, 54.4549789, -175.3055420, 175.4775238
40: -95.2996216, 46.9647598, -95.7878265, 47.2025604, -142.5021820, 142.7525940
41: -66.2731171, 39.2322464, -66.5748596, 39.4329453, -105.7060623, 105.8071060
42: -49.0356865, 43.1819839, -49.2878532, 43.5130386, -92.5487213, 92.4698334

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1759
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
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 614
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
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2951274, upper bound: 52.2952096
time: 68.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3359799, upper bound: 52.2952096
time: 70.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -61.9298553, 45.7031059, -61.9679604, 45.7107315, -107.6405792, 107.6710663
1: -39.3972549, 41.5953827, -39.4585190, 41.6449127, -81.0421677, 81.0539017
2: -36.3198242, 43.6078262, -36.3814545, 43.6084671, -79.9282913, 79.9892807
3: -44.2135849, 51.6314850, -44.2761574, 51.6789627, -95.8925400, 95.9076385
4: -51.8835945, 40.3421936, -51.8532066, 40.3283157, -92.2118988, 92.1954041
5: -46.0484009, 56.6406784, -46.1494141, 56.6757431, -102.7241364, 102.7900925
6: -67.4204712, 41.0416222, -67.4832611, 41.1888351, -108.6093063, 108.5248871
7: -56.4046898, 52.5617409, -56.5473366, 52.6957474, -109.1004333, 109.1090698
8: -46.5773621, 46.8231239, -46.6737289, 46.8580208, -93.4353714, 93.4968414
9: -48.8796577, 51.9098396, -48.9180984, 52.1142807, -100.9939423, 100.8279343
10: -78.4029388, 75.0783234, -78.4113541, 75.2023087, -153.6052399, 153.4896851
11: -79.8213959, 51.8439789, -79.7619553, 51.8046379, -131.6260376, 131.6059265
12: -73.7196655, 57.0667038, -73.7867584, 57.4302063, -131.1498718, 130.8534546
13: -70.2179489, 65.7322083, -70.2318497, 65.9922485, -136.2102051, 135.9640503
14: -105.7630768, 55.8279076, -105.9029160, 56.1419754, -161.9050446, 161.7308197
15: -58.1830139, 50.1506195, -58.2461166, 50.2456436, -108.4286575, 108.3967361
16: -82.3293457, 65.5577927, -82.3284149, 65.6121063, -147.9414520, 147.8862000
17: -118.2013550, 76.9262390, -118.2999954, 77.1954117, -195.3967590, 195.2262268
18: -68.6023254, 41.5619812, -68.6790924, 41.4859161, -110.0882263, 110.2410660
19: -59.6992035, 24.5889416, -59.7193108, 24.5012264, -84.2004318, 84.3082504
20: -53.8127785, 31.9414520, -53.8223648, 31.9425888, -85.7553711, 85.7638168
21: -72.0116577, 36.0471764, -71.9904785, 36.0083199, -108.0199661, 108.0376511
22: -81.2232819, 47.3512802, -81.4438248, 47.4600906, -128.6833649, 128.7951050
23: -54.5189972, 34.3424683, -54.5867615, 34.2525253, -88.7715225, 88.9292297
24: -63.8879051, 34.4398956, -64.0629807, 34.4513512, -98.3392563, 98.5028763
25: -59.6562195, 39.2137642, -59.7602768, 39.2439041, -98.9001236, 98.9740448
26: -91.9364624, 49.3980293, -92.0715332, 49.6460495, -141.5824890, 141.4695435
27: -67.5658264, 44.0197716, -67.7562103, 44.0223198, -111.5881424, 111.7759857
28: -56.2188797, 36.2996407, -56.3007927, 36.2688141, -92.4876862, 92.6004333
29: -81.0014038, 53.3295975, -81.1447601, 53.4537163, -134.4551086, 134.4743347
30: -67.6107635, 36.3981438, -67.7152252, 36.3832932, -103.9940567, 104.1133575
31: -62.1386948, 30.3152847, -62.2159157, 30.2454834, -92.3841782, 92.5312042
32: -65.1602554, 47.4610825, -65.1858978, 47.5777588, -112.7380142, 112.6469803
33: -98.8741760, 58.1596603, -98.9595871, 58.0861244, -156.9602966, 157.1192474
34: -84.2986603, 44.1925011, -84.4138794, 44.1958580, -128.4945221, 128.6063843
35: -79.8145905, 47.1100883, -79.8659668, 47.0827255, -126.8973083, 126.9760513
36: -81.9271545, 48.2122307, -81.9187927, 48.2168694, -130.1440125, 130.1310120
37: -114.6522980, 47.7703094, -114.7866745, 47.7781219, -162.4304199, 162.5569763
38: -101.3313675, 63.1961594, -101.4519501, 63.2669067, -164.5982361, 164.6481018
39: -121.5641098, 54.4596024, -121.5888443, 54.5191422, -176.0832520, 176.0484467
40: -95.9434662, 47.1857948, -96.1049576, 47.2478943, -143.1913605, 143.2907562
41: -66.5960388, 39.4838409, -66.7162094, 39.5509262, -106.1469574, 106.2000504
42: -49.3670845, 43.9179573, -49.3897934, 43.9189987, -93.2860870, 93.3077545

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 614
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
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
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
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2951274, upper bound: 52.2952179
time: 61.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3911822, upper bound: 52.2952179
time: 72.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -61.6554108, 45.4686966, -62.2120361, 45.9482346, -107.6036453, 107.6807251
1: -39.2445297, 41.4958191, -39.5917320, 41.7466354, -80.9911652, 81.0875549
2: -36.0667114, 43.4116287, -36.5410538, 43.8020477, -79.8687592, 79.9526825
3: -44.1000786, 51.4655495, -44.5433998, 51.9551544, -96.0552368, 96.0089493
4: -51.4907036, 40.1183167, -51.9879074, 40.4638710, -91.9545746, 92.1062241
5: -45.9076195, 56.4526596, -46.3895035, 56.9294357, -102.8370514, 102.8421555
6: -67.1670761, 40.6915016, -67.5903625, 41.1736603, -108.3407364, 108.2818527
7: -56.2236099, 52.3997345, -56.7494812, 52.7445183, -108.9681244, 109.1492157
8: -46.2142258, 46.5634842, -46.7433205, 46.9990654, -93.2132797, 93.3068008
9: -48.6462021, 51.5285072, -49.0975952, 52.1136475, -100.7598495, 100.6260986
10: -77.7691040, 74.4588165, -78.8308640, 75.4619217, -153.2310181, 153.2896729
11: -79.3372879, 51.3828506, -80.1638184, 52.0641975, -131.4014893, 131.5466614
12: -73.1696930, 56.3523026, -74.1986771, 57.5826073, -130.7523041, 130.5509796
13: -70.1441345, 65.2351456, -70.5616913, 66.1544495, -136.2985535, 135.7968445
14: -105.3166809, 55.4030724, -106.3711700, 56.3441048, -161.6607819, 161.7742462
15: -57.7446404, 49.8523636, -58.3613930, 50.3516235, -108.0962677, 108.2137604
16: -81.9434052, 65.0606079, -82.5519485, 65.6392059, -147.5826111, 147.6125488
17: -117.7584610, 76.2204285, -118.8032379, 77.4645767, -195.2230225, 195.0236664
18: -68.2389526, 41.5527954, -69.0296860, 41.8323784, -110.0713348, 110.5824738
19: -59.4765320, 24.6245155, -59.9769211, 24.7735100, -84.2500229, 84.6014404
20: -53.5800667, 31.8481369, -54.0768318, 32.1019211, -85.6819916, 85.9249573
21: -71.6978226, 36.0003510, -72.3489838, 36.3344345, -108.0322571, 108.3493347
22: -80.8843842, 47.2126312, -81.5150604, 47.7049599, -128.5893402, 128.7276917
23: -54.3451729, 34.2925797, -54.8342094, 34.4911880, -88.8363647, 89.1267853
24: -63.4005356, 34.3275604, -64.0563049, 34.5912781, -97.9917984, 98.3838654
25: -59.4334297, 39.1226006, -59.8754387, 39.4227409, -98.8561707, 98.9980392
26: -91.6378860, 49.2742233, -92.5034714, 50.0460663, -141.6839600, 141.7776947
27: -67.0269470, 43.8767014, -67.7317200, 44.1794968, -111.2064362, 111.6084137
28: -56.0293121, 36.2156029, -56.4498596, 36.4121323, -92.4414368, 92.6654663
29: -80.7262344, 53.2963905, -81.2414703, 53.7991447, -134.5253754, 134.5378418
30: -67.3946533, 36.2117767, -67.9940643, 36.5955887, -103.9902344, 104.2058411
31: -61.8135910, 30.3567181, -62.4823112, 30.5068474, -92.3204346, 92.8390198
32: -64.9201660, 47.1669960, -65.3080597, 47.5893936, -112.5095596, 112.4750519
33: -98.4727478, 57.7942543, -99.1758347, 58.4300270, -156.9027710, 156.9700928
34: -84.0758667, 43.9446373, -84.5968933, 44.3756142, -128.4514771, 128.5415344
35: -79.4885178, 46.7650223, -80.0755768, 47.3471718, -126.8356628, 126.8405991
36: -81.7461090, 48.0453911, -82.1242523, 48.3494568, -130.0955658, 130.1696472
37: -114.2024002, 47.5979004, -114.8539276, 47.9199905, -162.1223755, 162.4518280
38: -101.0159683, 63.0222435, -101.6245422, 63.4269180, -164.4428864, 164.6467896
39: -121.1365280, 54.3169632, -121.7790756, 54.7907257, -175.9272156, 176.0960388
40: -95.5000381, 47.0053177, -96.2095642, 47.4077797, -142.9078064, 143.2148743
41: -66.3808289, 39.3235779, -66.8163071, 39.6621017, -106.0429230, 106.1398773
42: -49.1166191, 43.4456062, -49.4854584, 44.0111275, -93.1277466, 92.9310532

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1742
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
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 744
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3400127
time: 68.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3400127
time: 66.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -62.1701622, 45.7679558, -62.4635658, 46.0282440, -108.1984100, 108.2315216
1: -39.5326385, 41.6508713, -39.7296181, 41.8117065, -81.3443451, 81.3804932
2: -36.5487709, 43.6526756, -36.7977066, 43.8489609, -80.3977356, 80.4503784
3: -44.4756012, 51.6936378, -44.7510300, 52.0412560, -96.5168533, 96.4446716
4: -52.1541405, 40.3978157, -52.3515739, 40.5379753, -92.6921082, 92.7493896
5: -46.2854004, 56.6990318, -46.5870628, 57.0104103, -103.2958069, 103.2860947
6: -67.5183487, 41.1250114, -67.7111969, 41.4027710, -108.9211121, 108.8362122
7: -56.5809212, 52.6108665, -56.9048004, 52.8463020, -109.4272232, 109.5156708
8: -46.7724609, 46.8872070, -47.0402412, 47.0723648, -93.8448257, 93.9274445
9: -49.0046196, 52.0408440, -49.1966515, 52.3899803, -101.3945923, 101.2374878
10: -78.5240479, 75.5771255, -78.9705353, 76.0996399, -154.6236572, 154.5476685
11: -79.9176025, 52.3154678, -80.2843781, 52.6036720, -132.5212708, 132.5998535
12: -73.7903900, 57.5256042, -74.2945099, 58.2485199, -132.0388947, 131.8201141
13: -70.4306870, 65.8650742, -70.6554260, 66.4571533, -136.8878326, 136.5204926
14: -105.9164581, 56.1967773, -106.5257645, 56.7945213, -162.7109833, 162.7225342
15: -58.3852158, 50.2480011, -58.6782913, 50.4666939, -108.8519058, 108.9262924
16: -82.4800949, 65.7820435, -82.7068176, 66.0460205, -148.5261230, 148.4888611
17: -118.3154755, 77.4925232, -118.9116211, 78.1903915, -196.5058594, 196.4041443
18: -68.7258759, 41.8137360, -69.2174988, 41.9513283, -110.6771851, 111.0312347
19: -59.7829285, 24.7829590, -60.0843658, 24.8463612, -84.6292725, 84.8673248
20: -53.8955193, 32.1010399, -54.1747360, 32.2400513, -86.1355743, 86.2757721
21: -72.1033630, 36.3316154, -72.4617767, 36.5166092, -108.6199646, 108.7933884
22: -81.3195496, 47.5617142, -81.7129288, 47.8566170, -129.1761627, 129.2746429
23: -54.5924721, 34.5393829, -54.9271507, 34.6161118, -89.2085648, 89.4665375
24: -63.9776459, 34.5374069, -64.3557587, 34.6470184, -98.6246643, 98.8931656
25: -59.7338562, 39.3571587, -60.0117035, 39.5176163, -99.2514725, 99.3688583
26: -92.0299225, 49.7602921, -92.6506042, 50.3192444, -142.3491669, 142.4108887
27: -67.7062302, 44.1253357, -68.0825272, 44.2373276, -111.9435577, 112.2078629
28: -56.2959900, 36.3953476, -56.5743713, 36.4737663, -92.7697601, 92.9697189
29: -81.0773239, 53.6184311, -81.3923721, 53.9710312, -135.0483551, 135.0108032
30: -67.6946411, 36.6199760, -68.1069946, 36.8056297, -104.5002747, 104.7269592
31: -62.2639618, 30.5003834, -62.6758919, 30.5731468, -92.8371124, 93.1762772
32: -65.2685547, 47.5782585, -65.4305115, 47.8193626, -113.0879211, 113.0087662
33: -99.2110519, 58.2413979, -99.5817108, 58.5262833, -157.7373352, 157.8231049
34: -84.5149078, 44.2695732, -84.8333893, 44.4627533, -128.9776611, 129.1029510
35: -80.1205368, 47.1802940, -80.4276581, 47.4293442, -127.5498810, 127.6079483
36: -82.1540375, 48.2735443, -82.3396759, 48.4189339, -130.5729675, 130.6132202
37: -114.8160858, 47.8684311, -115.1631088, 48.0106621, -162.8267517, 163.0315399
38: -101.5431137, 63.2821083, -101.8905487, 63.5262909, -165.0693970, 165.1726532
39: -121.8530884, 54.5220337, -122.1540756, 54.8549500, -176.7079926, 176.6761169
40: -96.1446838, 47.2258415, -96.5307007, 47.4544220, -143.5991058, 143.7565460
41: -66.7032623, 39.5787811, -66.9595795, 39.7898369, -106.4930878, 106.5383606
42: -49.4458580, 44.1831856, -49.5877953, 44.4312744, -93.8771286, 93.7709732

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 681
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
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
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 744
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1700
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
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
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
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
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
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3515444, upper bound: 52.2952179
time: 92.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4012924, upper bound: 52.3401102
time: 79.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -61.7951355, 45.6890106, -61.7908783, 45.7591705, -107.5543060, 107.4798813
1: -39.3679085, 41.6142082, -39.3610153, 41.6317902, -80.9996948, 80.9752197
2: -36.1880417, 43.6318779, -36.1710320, 43.6886253, -79.8766632, 79.8029099
3: -44.2230148, 51.7250519, -44.1170769, 51.7363205, -95.9593353, 95.8421326
4: -51.4878464, 40.2249146, -51.5397568, 40.3075905, -91.7954254, 91.7646713
5: -46.0906906, 56.7683334, -46.0053329, 56.7722664, -102.8629456, 102.7736588
6: -67.3149414, 40.9185295, -67.4160919, 41.0738525, -108.3887939, 108.3346024
7: -56.4901161, 52.6885376, -56.4586411, 52.7335739, -109.2236938, 109.1471710
8: -46.3578377, 46.7841263, -46.4126472, 46.9051285, -93.2629547, 93.1967773
9: -48.8720932, 51.7974396, -48.9729118, 51.8770027, -100.7490997, 100.7703476
10: -78.2183914, 74.6612854, -78.5328598, 74.6340637, -152.8524475, 153.1941528
11: -79.4726562, 51.2898903, -79.7325134, 51.3160553, -130.7886963, 131.0223999
12: -73.8161926, 56.7530518, -74.0591431, 56.8256607, -130.6418457, 130.8121948
13: -70.2751312, 65.5181046, -70.2834320, 65.7463074, -136.0214233, 135.8015289
14: -105.9459457, 55.7199936, -106.1028061, 55.7204666, -161.6664124, 161.8227997
15: -57.9031525, 50.0647621, -58.0453682, 50.1874008, -108.0905533, 108.1101303
16: -82.1042099, 65.1958466, -82.2713623, 65.2874908, -147.3916931, 147.4672089
17: -118.2846222, 76.4127655, -118.5026016, 76.5186539, -194.8032532, 194.9153748
18: -68.4468231, 41.5434456, -68.5979309, 41.4197350, -109.8665619, 110.1413727
19: -59.6047020, 24.5526047, -59.6860695, 24.4535828, -84.0582886, 84.2386780
20: -53.7266693, 31.8565998, -53.8030624, 31.8314400, -85.5581055, 85.6596527
21: -71.8607178, 35.9790611, -71.9707947, 35.8610153, -107.7217331, 107.9498444
22: -81.2387924, 47.4097862, -81.4346161, 47.3521538, -128.5909424, 128.8444061
23: -54.5062523, 34.2275658, -54.5609932, 34.1574249, -88.6636810, 88.7885590
24: -63.5928917, 34.3732185, -63.8158684, 34.4386940, -98.0315857, 98.1890869
25: -59.5777779, 39.2241364, -59.6968842, 39.1978226, -98.7756042, 98.9210205
26: -92.2454681, 49.6165619, -92.2532806, 49.4303589, -141.6758118, 141.8698425
27: -67.2519379, 43.9071159, -67.4602814, 44.0177231, -111.2696609, 111.3673859
28: -56.1811104, 36.2236938, -56.2353783, 36.2418747, -92.4229889, 92.4590759
29: -81.0378265, 53.4620476, -81.1579590, 53.3148842, -134.3527069, 134.6199951
30: -67.5951462, 36.2262688, -67.6607056, 36.2153702, -103.8104935, 103.8869781
31: -61.9682541, 30.3185196, -62.0949669, 30.2183495, -92.1865997, 92.4134827
32: -65.0236969, 47.2838593, -65.1244965, 47.3920555, -112.4157410, 112.4083405
33: -98.5649796, 57.8743439, -98.6186600, 58.0570526, -156.6220093, 156.4930115
34: -84.2368774, 44.0764923, -84.2322998, 44.1910706, -128.4279327, 128.3087921
35: -79.5643921, 46.8691902, -79.5706711, 47.0643730, -126.6287537, 126.4398422
36: -81.7917786, 48.1272125, -81.7719040, 48.1948547, -129.9866333, 129.8991089
37: -114.3880386, 47.7005272, -114.5593338, 47.7425652, -162.1306000, 162.2598572
38: -101.2254944, 63.1742859, -101.2490845, 63.2493591, -164.4748535, 164.4233551
39: -121.1914597, 54.4804344, -121.2879791, 54.5259323, -175.7173767, 175.7684021
40: -95.7100677, 47.2395363, -95.8381119, 47.3384247, -143.0484924, 143.0776520
41: -66.5650482, 39.4897461, -66.6184387, 39.5238609, -106.0889053, 106.1081848
42: -49.2212601, 43.4827843, -49.3407288, 43.5740776, -92.7953262, 92.8235168

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 964
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
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 809
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
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1503
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3436008
time: 80.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3436008
time: 88.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.3001137, 45.9836426, -62.0256538, 45.8359947, -108.1360931, 108.0092926
1: -39.6491928, 41.7712097, -39.4921265, 41.6945724, -81.3437653, 81.2633286
2: -36.6632500, 43.8645287, -36.4124527, 43.7331657, -80.3964081, 80.2769775
3: -44.6119308, 51.9501610, -44.3136215, 51.8174896, -96.4294052, 96.2637711
4: -52.1532288, 40.4972382, -51.8833275, 40.3790665, -92.5322876, 92.3805695
5: -46.4740334, 57.0064850, -46.1941757, 56.8500633, -103.3240967, 103.2006607
6: -67.6555023, 41.3074608, -67.5330048, 41.2555122, -108.9110107, 108.8404694
7: -56.8499794, 52.8750000, -56.6083641, 52.8228493, -109.6728210, 109.4833527
8: -46.9046326, 47.1064873, -46.6967659, 46.9757080, -93.8803406, 93.8032532
9: -49.2220383, 52.3109436, -49.0609398, 52.1489410, -101.3709793, 101.3718872
10: -78.9687347, 75.7710266, -78.6673355, 75.2472687, -154.2159729, 154.4383545
11: -79.9955750, 52.2047806, -79.8497162, 51.8253555, -131.8209229, 132.0544891
12: -74.4332504, 57.9191208, -74.1506882, 57.4656067, -131.8988647, 132.0698090
13: -70.5621490, 66.1404953, -70.3660126, 66.0447540, -136.6069031, 136.5065002
14: -106.5459061, 56.5269814, -106.2482910, 56.1661453, -162.7120361, 162.7752686
15: -58.4826279, 50.4218216, -58.3120079, 50.2996254, -108.7822418, 108.7338257
16: -82.5942993, 65.9094849, -82.4212799, 65.6733856, -148.2676849, 148.3307648
17: -118.8470688, 77.6739578, -118.6045380, 77.2073059, -196.0543518, 196.2784882
18: -68.9293518, 41.8009872, -68.7790680, 41.5284576, -110.4577942, 110.5800552
19: -59.9109268, 24.7065887, -59.7892380, 24.5189857, -84.4299164, 84.4958191
20: -54.0404854, 32.1060715, -53.8980637, 31.9624176, -86.0028992, 86.0041351
21: -72.2623291, 36.3143997, -72.0783615, 36.0375290, -108.2998505, 108.3927612
22: -81.6656952, 47.7423210, -81.6196671, 47.4995613, -129.1652527, 129.3619843
23: -54.7580185, 34.4680710, -54.6518021, 34.2743950, -89.0324097, 89.1198654
24: -64.1632385, 34.5692482, -64.1089172, 34.4916039, -98.6548462, 98.6781540
25: -59.8675041, 39.4484100, -59.8252029, 39.2889786, -99.1564636, 99.2736130
26: -92.6309280, 50.1283226, -92.3915176, 49.6965637, -142.3274841, 142.5198364
27: -67.9208069, 44.1628647, -67.8019257, 44.0716705, -111.9924622, 111.9647903
28: -56.4508743, 36.3967743, -56.3562851, 36.2996521, -92.7505264, 92.7530594
29: -81.3514404, 53.7816544, -81.2839508, 53.4834785, -134.8349152, 135.0656128
30: -67.8720093, 36.6225929, -67.7700500, 36.4113274, -104.2833405, 104.3926392
31: -62.4342384, 30.4572163, -62.2812080, 30.2783718, -92.7125931, 92.7384186
32: -65.3596039, 47.6939621, -65.2410965, 47.6082535, -112.9678497, 112.9350510
33: -99.2972565, 58.3168182, -99.0110703, 58.1495743, -157.4468384, 157.3278809
34: -84.6793823, 44.3968163, -84.4634933, 44.2734146, -128.9527893, 128.8603058
35: -80.1903839, 47.2798195, -79.9051971, 47.1447067, -127.3350830, 127.1850128
36: -82.2039490, 48.3400612, -81.9751587, 48.2639885, -130.4679413, 130.3152161
37: -114.9931107, 47.9577522, -114.8557892, 47.8295975, -162.8227081, 162.8135376
38: -101.7542267, 63.4317741, -101.5063324, 63.3424225, -165.0966339, 164.9380951
39: -121.8827820, 54.6847153, -121.6360245, 54.5884018, -176.4711914, 176.3207397
40: -96.3488083, 47.4577141, -96.1504593, 47.3803406, -143.7291565, 143.6081696
41: -66.8762817, 39.7038841, -66.7572021, 39.6154251, -106.4917068, 106.4610901
42: -49.5335159, 44.2095108, -49.4407845, 43.9609833, -93.4944916, 93.6502991

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 614
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1341
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
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3436008
time: 88.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3911822, upper bound: 52.3436008
time: 74.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -62.0301704, 45.7529831, -62.2732773, 46.0742035, -108.1043701, 108.0262604
1: -39.5016861, 41.6691055, -39.6272888, 41.7973099, -81.2989883, 81.2963867
2: -36.4123077, 43.6760864, -36.5744820, 43.9277840, -80.3400879, 80.2505646
3: -44.4818001, 51.7855301, -44.5827446, 52.0967445, -96.5785370, 96.3682709
4: -51.7527237, 40.2810478, -52.0227509, 40.5158348, -92.2685547, 92.3038025
5: -46.3257370, 56.8257141, -46.4358673, 57.1050758, -103.4308014, 103.2615662
6: -67.4130096, 40.9910278, -67.6412201, 41.2650299, -108.6780167, 108.6322479
7: -56.6666260, 52.7360497, -56.8112717, 52.8804893, -109.5471115, 109.5473175
8: -46.5495529, 46.8476295, -46.7696800, 47.1181259, -93.6676788, 93.6172943
9: -48.9943352, 51.9279099, -49.2422829, 52.1497536, -101.1440887, 101.1701965
10: -78.3382111, 75.1537628, -79.0891571, 75.5143127, -153.8525238, 154.2429199
11: -79.5702286, 51.7508011, -80.2526474, 52.0906601, -131.6608887, 132.0034485
12: -73.8858032, 57.2037277, -74.5646362, 57.6228714, -131.5086670, 131.7683411
13: -70.4838562, 65.6489639, -70.6976471, 66.2106171, -136.6944733, 136.3466187
14: -106.0958786, 56.0876732, -106.7202454, 56.3695908, -162.4654694, 162.8079224
15: -58.0912323, 50.1636467, -58.4502068, 50.4064026, -108.4976349, 108.6138535
16: -82.2569275, 65.4146118, -82.6469421, 65.7033920, -147.9603271, 148.0615540
17: -118.3965149, 76.9664612, -119.1100082, 77.4840469, -195.8805389, 196.0764313
18: -68.5675812, 41.7923393, -69.1347961, 41.8766365, -110.4442139, 110.9271393
19: -59.6865082, 24.7445526, -60.0485497, 24.7928982, -84.4794006, 84.7931061
20: -53.8085098, 32.0139236, -54.1540833, 32.1231117, -85.9316101, 86.1680069
21: -71.9508438, 36.2620621, -72.4390564, 36.3647003, -108.3155441, 108.7011032
22: -81.3308411, 47.6202583, -81.6943970, 47.7446632, -129.0755005, 129.3146515
23: -54.5792847, 34.4225082, -54.9005547, 34.5153275, -89.0946045, 89.3230591
24: -63.6800003, 34.4706726, -64.1062469, 34.6324272, -98.3124237, 98.5769196
25: -59.6525650, 39.3674278, -59.9439850, 39.4687653, -99.1213226, 99.3114090
26: -92.3353043, 49.9773102, -92.8274460, 50.0986519, -142.4339600, 142.8047485
27: -67.3892593, 44.0120926, -67.7824020, 44.2295303, -111.6187820, 111.7944946
28: -56.2569885, 36.3188324, -56.5069313, 36.4439697, -92.7009583, 92.8257599
29: -81.1093521, 53.7507439, -81.3988571, 53.8290062, -134.9383545, 135.1495972
30: -67.6790924, 36.4434547, -68.0503082, 36.6264725, -104.3055649, 104.4937515
31: -62.0923500, 30.5018692, -62.5519066, 30.5413074, -92.6336594, 93.0537720
32: -65.1318512, 47.3969727, -65.3648071, 47.6231003, -112.7549515, 112.7617722
33: -98.8975372, 57.9547462, -99.2306824, 58.4947701, -157.3923035, 157.1854248
34: -84.4518890, 44.1521187, -84.6484680, 44.4554405, -128.9073181, 128.8005829
35: -79.8647308, 46.9393768, -80.1187286, 47.4103394, -127.2750702, 127.0581055
36: -82.0150146, 48.1903648, -82.1838608, 48.3977776, -130.4127960, 130.3742218
37: -114.5486374, 47.7982483, -114.9278564, 47.9717598, -162.5203857, 162.7261047
38: -101.4358215, 63.2589378, -101.6821442, 63.5041771, -164.9400024, 164.9410858
39: -121.4719086, 54.5424957, -121.8384933, 54.8613663, -176.3332825, 176.3809814
40: -95.9085312, 47.2782936, -96.2572708, 47.5425797, -143.4511108, 143.5355682
41: -66.6722107, 39.5760231, -66.8584290, 39.7464409, -106.4186554, 106.4344482
42: -49.3013420, 43.7376442, -49.5378494, 44.0614166, -93.3627548, 93.2754898

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1598
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
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1729
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
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 707
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
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 744
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
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1533
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
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1428
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 1359
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
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3070972, upper bound: 52.3857433
time: 1581.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3857433
time: 77.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 1661.47 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.2951274, upper bound: 52.2952096
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3359799, upper bound: 52.2952096
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.2951274, upper bound: 52.2952179
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3911822, upper bound: 52.2952179
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3400127
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3400127
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3515444, upper bound: 52.2952179
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.4012924, upper bound: 52.3401102
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3436008
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3436008
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.2951274, upper bound: 52.3436008
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3911822, upper bound: 52.3436008
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3070972, upper bound: 52.3857433
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1661.47
Output dim: 2, lower bound: -52.3359799, upper bound: 52.3857433
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3741897
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4381855
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4893945
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3740869
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.4772059, upper bound: 52.3741897
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4381855
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4253881
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.3378656, upper bound: 52.4255597
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.4342985, upper bound: 52.4891413
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1661.47
Output dim: 2, lower bound: -52.4893949, upper bound: 52.4893945

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 82.72 + 7860.67 = 7943.39 seconds
