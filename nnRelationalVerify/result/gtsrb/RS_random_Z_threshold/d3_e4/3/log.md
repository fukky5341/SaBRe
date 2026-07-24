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
execution time: IAR + RelationalAnalysis = 2.92 + 75.98 = 78.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -52.5034420, upper bound: 52.5034420

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1343

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 602

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4836034, upper bound: 52.4896087
time: 71.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4836034, upper bound: 52.4836034
time: 90.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 162.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 162.20
Output dim: 2, lower bound: -52.4836034, upper bound: 52.4896087
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 162.20
Output dim: 2, lower bound: -52.4836034, upper bound: 52.4836034

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4704631, upper bound: 52.4893995
time: 69.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4833955, upper bound: 52.4764551
time: 73.62 seconds

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
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 851

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1621

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4875233, upper bound: 52.4496839
time: 101.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4557305, upper bound: 52.4815178
time: 75.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 179.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 179.43
Output dim: 2, lower bound: -52.4704631, upper bound: 52.4893995
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 179.43
Output dim: 2, lower bound: -52.4833955, upper bound: 52.4764551
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 179.43
Output dim: 2, lower bound: -52.4875233, upper bound: 52.4496839
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 179.43
Output dim: 2, lower bound: -52.4557305, upper bound: 52.4815178

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

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 679

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4656263, upper bound: 52.4251931
time: 59.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4062006, upper bound: 52.4845633
time: 68.95 seconds

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4396981, upper bound: 52.4332419
time: 67.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4401382, upper bound: 52.4328021
time: 65.58 seconds

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1294

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 861

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4694228, upper bound: 52.4494401
time: 85.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4872808, upper bound: 52.4315135
time: 71.58 seconds

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
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4427731, upper bound: 52.4741106
time: 59.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4484970, upper bound: 52.4684695
time: 74.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 137.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4656263, upper bound: 52.4251931
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4062006, upper bound: 52.4845633
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4396981, upper bound: 52.4332419
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4401382, upper bound: 52.4328021
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4694228, upper bound: 52.4494401
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4872808, upper bound: 52.4315135
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4427731, upper bound: 52.4741106
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 137.19
Output dim: 2, lower bound: -52.4484970, upper bound: 52.4684695

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

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 842

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4639979, upper bound: 52.4058386
time: 88.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4461204, upper bound: 52.4235974
time: 62.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1729

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 948

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4052653, upper bound: 52.4826124
time: 64.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4042462, upper bound: 52.4836258
time: 62.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1018

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1605

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4687495, upper bound: 52.4320449
time: 302.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4406731, upper bound: 52.4466261
time: 66.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1679

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4868516, upper bound: 52.4249633
time: 71.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4807740, upper bound: 52.4310831
time: 81.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1018

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4340583, upper bound: 52.4736824
time: 75.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4423455, upper bound: 52.4653909
time: 81.41 seconds

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

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 630

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4165756, upper bound: 52.4332114
time: 69.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4132079, upper bound: 52.4367838
time: 83.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 156.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4639979, upper bound: 52.4058386
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4461204, upper bound: 52.4235974
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4052653, upper bound: 52.4826124
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4042462, upper bound: 52.4836258
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4687495, upper bound: 52.4320449
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4406731, upper bound: 52.4466261
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4868516, upper bound: 52.4249633
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4807740, upper bound: 52.4310831
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4340583, upper bound: 52.4736824
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4423455, upper bound: 52.4653909
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4165756, upper bound: 52.4332114
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 156.05
Output dim: 2, lower bound: -52.4132079, upper bound: 52.4367838

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

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1721

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4551045, upper bound: 52.4033265
time: 68.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4615525, upper bound: 52.3968525
time: 75.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 618

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3960797, upper bound: 52.4741161
time: 70.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3969862, upper bound: 52.4720236
time: 83.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 703

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 925

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3823082, upper bound: 52.4834872
time: 71.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4041091, upper bound: 52.4617445
time: 79.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1487

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4678813, upper bound: 52.4314872
time: 86.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4681263, upper bound: 52.4312319
time: 73.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4821866, upper bound: 52.4069975
time: 79.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4681284, upper bound: 52.4203843
time: 90.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1633

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4805413, upper bound: 52.4280585
time: 80.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4777158, upper bound: 52.4308532
time: 81.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4254778, upper bound: 52.4735912
time: 74.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4339683, upper bound: 52.4650972
time: 87.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4413125, upper bound: 52.4623808
time: 87.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4393242, upper bound: 52.4643693
time: 80.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 171.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4551045, upper bound: 52.4033265
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4615525, upper bound: 52.3968525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.3960797, upper bound: 52.4741161
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.3969862, upper bound: 52.4720236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.3823082, upper bound: 52.4834872
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4041091, upper bound: 52.4617445
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4678813, upper bound: 52.4314872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4681263, upper bound: 52.4312319
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4821866, upper bound: 52.4069975
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4681284, upper bound: 52.4203843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4805413, upper bound: 52.4280585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4777158, upper bound: 52.4308532
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4254778, upper bound: 52.4735912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4339683, upper bound: 52.4650972
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4413125, upper bound: 52.4623808
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 171.09
Output dim: 2, lower bound: -52.4393242, upper bound: 52.4643693

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

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1622

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4534059, upper bound: 52.3838583
time: 75.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4350794, upper bound: 52.4018096
time: 72.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1766

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1291

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4581911, upper bound: 52.3934730
time: 79.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4581911, upper bound: 52.3934730
time: 79.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1287

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3934949, upper bound: 52.4715565
time: 70.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3934949, upper bound: 52.4715565
time: 70.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1617

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3967892, upper bound: 52.4714014
time: 77.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3963768, upper bound: 52.4718247
time: 74.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 941

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3759592, upper bound: 52.4800879
time: 68.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3788787, upper bound: 52.4770910
time: 76.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 956

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3969320, upper bound: 52.4615263
time: 96.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4038747, upper bound: 52.4545112
time: 69.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1352

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4533502, upper bound: 52.4291810
time: 79.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4654929, upper bound: 52.4170629
time: 109.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 879

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1347

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4675483, upper bound: 52.4276713
time: 75.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4646059, upper bound: 52.4306007
time: 82.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1566

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4783658, upper bound: 52.4054947
time: 80.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4666635, upper bound: 52.4031171
time: 66.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 879

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1001

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4335463, upper bound: 52.4201367
time: 71.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4678823, upper bound: 52.3858510
time: 63.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 989

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1515

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4803292, upper bound: 52.4266372
time: 70.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4791284, upper bound: 52.4278493
time: 67.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1354

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1352

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4769643, upper bound: 52.4305600
time: 64.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4774191, upper bound: 52.4300935
time: 70.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 999

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3936285, upper bound: 52.4536658
time: 60.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4056325, upper bound: 52.4417152
time: 68.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 827

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4198549, upper bound: 52.4646663
time: 69.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4335593, upper bound: 52.4510030
time: 72.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1285

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4140718, upper bound: 52.4622289
time: 78.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4411606, upper bound: 52.4351982
time: 78.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1283

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4376927, upper bound: 52.4626456
time: 71.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4376059, upper bound: 52.4627313
time: 73.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 147.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4534059, upper bound: 52.3838583
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4350794, upper bound: 52.4018096
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4581911, upper bound: 52.3934730
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4581911, upper bound: 52.3934730
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3934949, upper bound: 52.4715565
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3934949, upper bound: 52.4715565
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3967892, upper bound: 52.4714014
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3963768, upper bound: 52.4718247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3759592, upper bound: 52.4800879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3788787, upper bound: 52.4770910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3969320, upper bound: 52.4615263
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4038747, upper bound: 52.4545112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4533502, upper bound: 52.4291810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4654929, upper bound: 52.4170629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4675483, upper bound: 52.4276713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4646059, upper bound: 52.4306007
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4783658, upper bound: 52.4054947
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4666635, upper bound: 52.4031171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4335463, upper bound: 52.4201367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4678823, upper bound: 52.3858510
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4803292, upper bound: 52.4266372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4791284, upper bound: 52.4278493
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4769643, upper bound: 52.4305600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4774191, upper bound: 52.4300935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.3936285, upper bound: 52.4536658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4056325, upper bound: 52.4417152
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4198549, upper bound: 52.4646663
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4335593, upper bound: 52.4510030
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4140718, upper bound: 52.4622289
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4411606, upper bound: 52.4351982
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4376927, upper bound: 52.4626456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 147.40
Output dim: 2, lower bound: -52.4376059, upper bound: 52.4627313

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4506482, upper bound: 52.3719851
time: 99.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4415124, upper bound: 52.3811349
time: 95.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4539170, upper bound: 52.3903740
time: 70.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4552241, upper bound: 52.3889940
time: 103.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1490

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1359

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4559612, upper bound: 52.3925088
time: 75.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4572498, upper bound: 52.3912162
time: 76.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 989

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3929453, upper bound: 52.4567303
time: 75.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3785839, upper bound: 52.4709855
time: 68.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1727

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1596

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3933209, upper bound: 52.4622382
time: 72.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3831458, upper bound: 52.4713826
time: 71.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1366

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1583

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3898782, upper bound: 52.4711074
time: 72.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3964877, upper bound: 52.4645035
time: 75.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1687

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3955995, upper bound: 52.4621085
time: 83.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3866640, upper bound: 52.4710507
time: 59.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1343
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1396
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 1286
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1285
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1294
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1283
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1291
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1708

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1637

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3752730, upper bound: 52.4585715
time: 68.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.3540284, upper bound: 52.4794068
time: 68.89 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 139.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4506482, upper bound: 52.3719851
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4415124, upper bound: 52.3811349
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4539170, upper bound: 52.3903740
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4552241, upper bound: 52.3889940
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4559612, upper bound: 52.3925088
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.4572498, upper bound: 52.3912162
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3929453, upper bound: 52.4567303
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3785839, upper bound: 52.4709855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3933209, upper bound: 52.4622382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3831458, upper bound: 52.4713826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3898782, upper bound: 52.4711074
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3964877, upper bound: 52.4645035
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3955995, upper bound: 52.4621085
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3866640, upper bound: 52.4710507
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3752730, upper bound: 52.4585715
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 139.66
Output dim: 2, lower bound: -52.3540284, upper bound: 52.4794068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.3788787, upper bound: 52.4770910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.3969320, upper bound: 52.4615263
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4038747, upper bound: 52.4545112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4533502, upper bound: 52.4291810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4654929, upper bound: 52.4170629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4675483, upper bound: 52.4276713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4646059, upper bound: 52.4306007
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4783658, upper bound: 52.4054947
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4666635, upper bound: 52.4031171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4678823, upper bound: 52.3858510
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4803292, upper bound: 52.4266372
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4791284, upper bound: 52.4278493
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4769643, upper bound: 52.4305600
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4774191, upper bound: 52.4300935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.3936285, upper bound: 52.4536658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4198549, upper bound: 52.4646663
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4335593, upper bound: 52.4510030
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4140718, upper bound: 52.4622289
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4376927, upper bound: 52.4626456
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 139.66
Output dim: 2, lower bound: -52.4376059, upper bound: 52.4627313

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 78.90 + 7152.00 = 7230.90 seconds
