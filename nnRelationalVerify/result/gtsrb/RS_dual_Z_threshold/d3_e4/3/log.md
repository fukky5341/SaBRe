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
execution time: IAR + RelationalAnalysis = 2.95 + 77.35 = 80.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -52.5034420, upper bound: 52.5034420

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4304952, upper bound: 52.5007533
time: 95.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.5007533, upper bound: 52.4304952
time: 73.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 169.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 169.23
Output dim: 2, lower bound: -52.4304952, upper bound: 52.5007533
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 169.23
Output dim: 2, lower bound: -52.5007533, upper bound: 52.4304952

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3989100, upper bound: 52.4988609
time: 67.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4261165, upper bound: 52.4521347
time: 89.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4521347, upper bound: 52.4261165
time: 71.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4988609, upper bound: 52.3989100
time: 65.06 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 139.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 139.39
Output dim: 2, lower bound: -52.3989100, upper bound: 52.4988609
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 139.39
Output dim: 2, lower bound: -52.4261165, upper bound: 52.4521347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 139.39
Output dim: 2, lower bound: -52.4521347, upper bound: 52.4261165
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 139.39
Output dim: 2, lower bound: -52.4988609, upper bound: 52.3989100

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3359618, upper bound: 52.4889830
time: 67.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3885380, upper bound: 52.4367659
time: 74.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3612612, upper bound: 52.4428579
time: 72.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4159517, upper bound: 52.3938143
time: 71.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3938143, upper bound: 52.4159517
time: 55.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4428579, upper bound: 52.3612612
time: 74.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4367659, upper bound: 52.3885381
time: 108.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4889830, upper bound: 52.3359618
time: 69.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 180.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.3359618, upper bound: 52.4889830
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.3885380, upper bound: 52.4367659
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.3612612, upper bound: 52.4428579
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.4159517, upper bound: 52.3938143
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.3938143, upper bound: 52.4159517
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.4428579, upper bound: 52.3612612
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.4367659, upper bound: 52.3885381
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 180.29
Output dim: 2, lower bound: -52.4889830, upper bound: 52.3359618

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2788693, upper bound: 52.4856541
time: 67.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3302463, upper bound: 52.4091446
time: 72.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3824336, upper bound: 52.3302463
time: 60.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4856541, upper bound: 52.2788693
time: 67.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 130.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 130.56
Output dim: 2, lower bound: -52.2788693, upper bound: 52.4856541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 130.56
Output dim: 2, lower bound: -52.3302463, upper bound: 52.4091446
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 130.56
Output dim: 2, lower bound: -52.3824336, upper bound: 52.3302463
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 130.56
Output dim: 2, lower bound: -52.4856541, upper bound: 52.2788693

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.2310107, upper bound: 52.4840440
time: 67.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.2769662, upper bound: 52.4395384
time: 65.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4395384, upper bound: 52.2769662
time: 82.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4840440, upper bound: 52.2310107
time: 67.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 152.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 152.59
Output dim: 2, lower bound: -52.2310107, upper bound: 52.4840440
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 152.59
Output dim: 2, lower bound: -52.2769662, upper bound: 52.4395384
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 152.59
Output dim: 2, lower bound: -52.4395384, upper bound: 52.2769662
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 152.59
Output dim: 2, lower bound: -52.4840440, upper bound: 52.2310107

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.1771985, upper bound: 52.4796606
time: 67.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.2229221, upper bound: 52.4105227
time: 67.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3617779, upper bound: 52.2229221
time: 73.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4796606, upper bound: 52.1771985
time: 65.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 141.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 141.80
Output dim: 2, lower bound: -52.1771985, upper bound: 52.4796606
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 141.80
Output dim: 2, lower bound: -52.2229221, upper bound: 52.4105227
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 141.80
Output dim: 2, lower bound: -52.3617779, upper bound: 52.2229221
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 141.80
Output dim: 2, lower bound: -52.4796606, upper bound: 52.1771985

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.1294666, upper bound: 52.4770313
time: 78.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.1740343, upper bound: 52.4184915
time: 61.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4184915, upper bound: 52.1740343
time: 69.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4770313, upper bound: 52.1294666
time: 75.82 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 147.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 147.89
Output dim: 2, lower bound: -52.1294666, upper bound: 52.4770313
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 147.89
Output dim: 2, lower bound: -52.1740343, upper bound: 52.4184915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 147.89
Output dim: 2, lower bound: -52.4184915, upper bound: 52.1740343
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 147.89
Output dim: 2, lower bound: -52.4770313, upper bound: 52.1294666

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.1163016, upper bound: 52.4222638
time: 74.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.0749390, upper bound: 52.4633165
time: 63.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4633165, upper bound: 52.0749390
time: 102.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4222638, upper bound: 52.1163016
time: 69.95 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 175.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 175.27
Output dim: 2, lower bound: -52.1163016, upper bound: 52.4222638
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 175.27
Output dim: 2, lower bound: -52.0749390, upper bound: 52.4633165
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 175.27
Output dim: 2, lower bound: -52.4633165, upper bound: 52.0749390
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 175.27
Output dim: 2, lower bound: -52.4222638, upper bound: 52.1163016

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.0395058, upper bound: 52.3721300
time: 78.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.0199363, upper bound: 52.4290559
time: 65.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -63.3322601, 46.2682877, -63.3322601, 46.2682877, -109.6005402, 109.6005402
1: -40.2499237, 42.0159874, -40.2499237, 42.0159874, -82.2659073, 82.2658997
2: -37.6992912, 44.0498428, -37.6992912, 44.0498428, -81.7491302, 81.7491302
3: -45.6764526, 52.2865143, -45.6764526, 52.2865143, -97.9629669, 97.9629517
4: -53.3824730, 40.7526131, -53.3824730, 40.7526131, -94.1350861, 94.1350861
5: -47.5138054, 57.3164406, -47.5138054, 57.3164406, -104.8302460, 104.8302460
6: -68.0753326, 42.1878357, -68.0753326, 42.1878357, -110.2631683, 110.2631607
7: -57.6719894, 53.2222786, -57.6719894, 53.2222786, -110.8942642, 110.8942642
8: -47.9569931, 47.3655663, -47.9569931, 47.3655663, -95.3225555, 95.3225555
9: -49.6739120, 53.1676331, -49.6739120, 53.1676331, -102.8415451, 102.8415451
10: -79.4868164, 77.8195038, -79.4868164, 77.8195038, -157.3063202, 157.3063202
11: -80.4269714, 53.9462776, -80.4269714, 53.9462776, -134.3732300, 134.3732452
12: -74.7787018, 60.1630783, -74.7787018, 60.1630783, -134.9417725, 134.9417725
13: -71.1126099, 66.9305878, -71.1126099, 66.9305878, -138.0431976, 138.0431976
14: -107.2073135, 57.9945145, -107.2073135, 57.9945145, -165.2018127, 165.2018280
15: -59.6691780, 50.8379517, -59.6691780, 50.8379517, -110.5071259, 110.5071259
16: -83.2061920, 67.0601807, -83.2061920, 67.0601807, -150.2663727, 150.2663727
17: -119.3261948, 79.9495697, -119.3261948, 79.9495697, -199.2757568, 199.2757568
18: -69.5507965, 42.5489273, -69.5507965, 42.5489273, -112.0997086, 112.0997162
19: -60.2952843, 25.2580280, -60.2952843, 25.2580280, -85.5533066, 85.5533142
20: -54.3931046, 32.6802292, -54.3931046, 32.6802292, -87.0733337, 87.0733261
21: -72.6607971, 37.2185631, -72.6607971, 37.2185631, -109.8793640, 109.8793640
22: -82.3638382, 48.4710922, -82.3638382, 48.4710922, -130.8349304, 130.8349304
23: -55.0878067, 35.0359001, -55.0878067, 35.0359001, -90.1237030, 90.1237030
24: -64.8367157, 34.8513069, -64.8367157, 34.8513069, -99.6880188, 99.6880188
25: -60.3260689, 39.9170074, -60.3260689, 39.9170074, -100.2430573, 100.2430649
26: -93.1423798, 51.5078735, -93.1423798, 51.5078735, -144.6502380, 144.6502533
27: -68.7734070, 44.4622612, -68.7734070, 44.4622612, -113.2356567, 113.2356644
28: -56.7897606, 36.7053833, -56.7897606, 36.7053833, -93.4951477, 93.4951477
29: -81.8284225, 54.7636337, -81.8284225, 54.7636337, -136.5920563, 136.5920563
30: -68.2478333, 37.4534225, -68.2478333, 37.4534225, -105.7012482, 105.7012482
31: -63.0613937, 30.9135685, -63.0613937, 30.9135685, -93.9749603, 93.9749603
32: -65.7900925, 48.4671860, -65.7900925, 48.4671860, -114.2572708, 114.2572784
33: -100.5602570, 58.6897125, -100.5602570, 58.6897125, -159.2499695, 159.2499695
34: -85.4649658, 44.7364731, -85.4649658, 44.7364731, -130.2014465, 130.2014313
35: -81.2894821, 47.5783920, -81.2894821, 47.5783920, -128.8678741, 128.8678741
36: -82.8692322, 48.6446877, -82.8692322, 48.6446877, -131.5139160, 131.5139160
37: -115.7810059, 48.3521652, -115.7810059, 48.3521652, -164.1331787, 164.1331787
38: -102.5896149, 63.8586655, -102.5896149, 63.8586655, -166.4482727, 166.4482727
39: -122.9429550, 54.9620819, -122.9429550, 54.9620819, -177.9050293, 177.9050293
40: -97.2474670, 47.6734161, -97.2474670, 47.6734161, -144.9208832, 144.9208832
41: -67.3548355, 40.3282013, -67.3548355, 40.3282013, -107.6830368, 107.6830368
42: -49.8824310, 45.4829788, -49.8824310, 45.4829788, -95.3654099, 95.3654099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=372, inp2_unstable=372, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1286

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4290559, upper bound: 52.0199363
time: 62.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3721300, upper bound: 52.0395058
time: 72.88 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 137.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 137.95
Output dim: 2, lower bound: -52.0395058, upper bound: 52.3721300
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 137.95
Output dim: 2, lower bound: -52.0199363, upper bound: 52.4290559
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 137.95
Output dim: 2, lower bound: -52.4290559, upper bound: 52.0199363
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 137.95
Output dim: 2, lower bound: -52.3721300, upper bound: 52.0395058

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 80.30 + 2801.87 = 2882.17 seconds
