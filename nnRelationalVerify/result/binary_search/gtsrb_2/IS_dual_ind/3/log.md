## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 52.4281605764
Search space: {k/256.0 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 2.90 + 88.21 = 91.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -61.8292597, upper bound: 61.8292597


# Binary Search by BASE starts (time budget: 17908.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=81.74913024902344
rel_dist={2: [-56.40659389343212, 56.40659388793267]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=81.74913024902344
rel_dist={2: [-52.50344204029277, 52.50344204220282]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=81.74913024902344
rel_dist={2: [-49.099917939237244, 49.099917941883625]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=81.74913024902344
rel_dist={2: [-50.88808347724346, 50.88808348062065]}

## Binary Search Result
Binary search time: 337.31 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 17571.58 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

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
Output dim: 2, lower bound: -57.3618347, upper bound: 57.4481367
time: 85.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3618347, upper bound: 57.4481364
time: 70.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 156.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 156.27
Output dim: 2, lower bound: -57.3618347, upper bound: 57.4481367
IS_A2, status: Status.UNKNOWN, split count: 1, time: 156.27
Output dim: 2, lower bound: -57.3618347, upper bound: 57.4481364

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -63.0016670, 46.1784630, -63.2562599, 46.2479095, -109.2495728, 109.4347229
1: -40.0608215, 41.9438629, -40.2071495, 41.9995918, -82.0604095, 82.1510162
2: -37.3429146, 43.9967194, -37.6187897, 44.0377960, -81.3807068, 81.6155090
3: -45.3487587, 52.1926460, -45.6022148, 52.2652206, -97.6139679, 97.7948608
4: -52.9470558, 40.6721992, -53.2840042, 40.7344170, -93.6814728, 93.9561996
5: -47.1742783, 57.2239685, -47.4365845, 57.2954903, -104.4697723, 104.6605530
6: -67.9469528, 41.8579330, -68.0462570, 42.1119652, -110.0589142, 109.9041901
7: -57.4120445, 53.1295624, -57.6129417, 53.2011261, -110.6131592, 110.7425003
8: -47.5625572, 47.2853432, -47.8676529, 47.3474007, -94.9099579, 95.1529922
9: -49.5510902, 52.8360023, -49.6461487, 53.0928230, -102.6439056, 102.4821472
10: -79.3242340, 77.1661530, -79.4501495, 77.6720276, -156.9962616, 156.6162872
11: -80.2767029, 53.3444786, -80.3928833, 53.8103714, -134.0870667, 133.7373657
12: -74.6675873, 59.2950630, -74.7535629, 59.9670792, -134.6346741, 134.0486145
13: -71.0004272, 66.5607605, -71.0871277, 66.8462524, -137.8466644, 137.6478882
14: -107.0077667, 57.4680252, -107.1621017, 57.8727264, -164.8804932, 164.6301270
15: -59.2739868, 50.7019501, -59.5773926, 50.8071632, -110.0811462, 110.2793427
16: -83.0029984, 66.6599503, -83.1603470, 66.9682007, -149.9711914, 149.8202820
17: -119.1782837, 79.0953064, -119.2927170, 79.7551422, -198.9333801, 198.3880157
18: -69.3224792, 42.3745041, -69.4987030, 42.5089340, -111.8314056, 111.8731995
19: -60.1685791, 25.1261559, -60.2663612, 25.2281532, -85.3967285, 85.3925171
20: -54.2787437, 32.4904060, -54.3671761, 32.6373062, -86.9160385, 86.8575745
21: -72.5295181, 36.9514503, -72.6309509, 37.1584320, -109.6879425, 109.5823975
22: -82.1464233, 48.2740059, -82.3135681, 48.4268570, -130.5732727, 130.5875549
23: -54.9721603, 34.8977737, -55.0616684, 35.0046158, -89.9767761, 89.9594421
24: -64.5143280, 34.7826767, -64.7635651, 34.8358040, -99.3501282, 99.5462418
25: -60.1498032, 39.7958908, -60.2855186, 39.8896637, -100.0394669, 100.0814056
26: -92.9877014, 51.0992432, -93.1070786, 51.4158211, -144.4035034, 144.2063293
27: -68.4123840, 44.3966713, -68.6911621, 44.4474258, -112.8598099, 113.0878296
28: -56.6728668, 36.6275101, -56.7631760, 36.6876602, -93.3605270, 93.3906860
29: -81.6733551, 54.4776573, -81.7931290, 54.6988945, -136.3722382, 136.2707825
30: -68.1156845, 37.1841164, -68.2177582, 37.3922577, -105.5079422, 105.4018707
31: -62.8120842, 30.8302631, -63.0042839, 30.8946419, -93.7067261, 93.8345490
32: -65.6613159, 48.1253128, -65.7608643, 48.3897705, -114.0510712, 113.8861771
33: -100.1179428, 58.5664062, -100.4602280, 58.6617012, -158.7796326, 159.0266418
34: -85.2263031, 44.6338501, -85.4109039, 44.7132492, -129.9395447, 130.0447388
35: -80.9530182, 47.4830894, -81.2133255, 47.5568428, -128.5098267, 128.6964111
36: -82.7201691, 48.5291252, -82.8353348, 48.6184464, -131.3386230, 131.3644562
37: -115.5018463, 48.2389297, -115.7176132, 48.3265839, -163.8284302, 163.9565430
38: -102.3634338, 63.6911469, -102.5382919, 63.8199120, -166.1833496, 166.2294312
39: -122.6005096, 54.8789597, -122.8643417, 54.9433136, -177.5438232, 177.7433014
40: -96.9164505, 47.6048622, -97.1723099, 47.6577415, -144.5741882, 144.7771759
41: -67.1906281, 40.1013680, -67.3178101, 40.2763977, -107.4670258, 107.4191666
42: -49.7730141, 45.0194016, -49.8576965, 45.3777924, -95.1508026, 94.8770981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
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
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1670
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
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1679
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
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1403
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
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 680

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3432837, upper bound: 57.2949341
time: 72.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3579696, upper bound: 57.4442936
time: 72.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -63.3884659, 46.4092941, -63.3185577, 46.2646484, -109.6531143, 109.7278519
1: -40.2785645, 42.0561752, -40.2395706, 42.0121727, -82.2907410, 82.2957458
2: -37.7138939, 44.2348862, -37.6872597, 44.0474396, -81.7613220, 81.9221497
3: -45.6939659, 52.4097023, -45.6642227, 52.2827415, -97.9767075, 98.0739212
4: -53.4087982, 40.9603653, -53.3677444, 40.7487984, -94.1575928, 94.3281021
5: -47.5300598, 57.5281029, -47.5019073, 57.3128510, -104.8428879, 105.0300064
6: -68.2390747, 42.1814842, -68.0697174, 42.1635284, -110.4026031, 110.2512054
7: -57.7241211, 53.2586327, -57.6593399, 53.2154694, -110.9395905, 110.9179611
8: -47.9823532, 47.5534782, -47.9432869, 47.3617935, -95.3441467, 95.4967651
9: -49.7788239, 53.2098503, -49.6682281, 53.1563835, -102.9352036, 102.8780823
10: -79.8044739, 77.8485413, -79.4799423, 77.7984772, -157.6029358, 157.3284912
11: -80.6970978, 53.9490051, -80.4208832, 53.9268379, -134.6239166, 134.3698883
12: -75.2563324, 60.1641998, -74.7738647, 60.1365356, -135.3928680, 134.9380646
13: -71.2097778, 67.0046844, -71.1070099, 66.9153442, -138.1251221, 138.1116943
14: -107.4735489, 57.9991798, -107.1976089, 57.9785805, -165.4521332, 165.1967773
15: -59.6784286, 51.0344505, -59.6356735, 50.8327179, -110.5111237, 110.6701202
16: -83.3799133, 67.0744476, -83.1979294, 67.0391693, -150.4190826, 150.2723694
17: -119.6977081, 79.9486084, -119.3188934, 79.9229279, -199.6206360, 199.2675018
18: -69.6526642, 42.5886497, -69.5300751, 42.5396194, -112.1922836, 112.1187286
19: -60.4160500, 25.2779121, -60.2886314, 25.2527695, -85.6688232, 85.5665436
20: -54.5291977, 32.6928902, -54.3886795, 32.6732483, -87.2024460, 87.0815659
21: -72.8720551, 37.2309341, -72.6550674, 37.2095871, -110.0816345, 109.8860016
22: -82.3923416, 48.5531044, -82.3377762, 48.4640236, -130.8563538, 130.8908844
23: -55.1852112, 35.0653648, -55.0833549, 35.0305176, -90.2157288, 90.1487045
24: -64.8885498, 34.9694977, -64.8253403, 34.8475037, -99.7360535, 99.7948303
25: -60.3670120, 39.9802551, -60.3117447, 39.9119949, -100.2789993, 100.2919922
26: -93.2987366, 51.5121956, -93.1352921, 51.4862900, -144.7850037, 144.6474915
27: -68.8355484, 44.5341835, -68.7598953, 44.4585152, -113.2940598, 113.2940826
28: -56.8597832, 36.7313995, -56.7839394, 36.7007256, -93.5605087, 93.5153275
29: -81.8847046, 54.7821350, -81.8121262, 54.7512169, -136.6359100, 136.5942535
30: -68.3390198, 37.4886322, -68.2415619, 37.4436531, -105.7826691, 105.7301941
31: -63.1854401, 30.9387474, -63.0519676, 30.9086151, -94.0940552, 93.9907150
32: -65.9957962, 48.4745941, -65.7844238, 48.4553413, -114.4511414, 114.2590179
33: -100.6208496, 58.8414917, -100.5455627, 58.6834908, -159.3043365, 159.3870544
34: -85.5063934, 44.8319778, -85.4562149, 44.7306519, -130.2370453, 130.2881775
35: -81.3342209, 47.6985779, -81.2776413, 47.5744057, -128.9086151, 128.9762115
36: -82.9317322, 48.6838875, -82.8590012, 48.6380615, -131.5697937, 131.5428925
37: -115.8673401, 48.4071426, -115.7664032, 48.3473358, -164.2146606, 164.1735382
38: -102.6810760, 63.9082031, -102.5773697, 63.8500214, -166.5310822, 166.4855652
39: -123.0573502, 55.0508804, -122.9310532, 54.9572067, -178.0145569, 177.9819336
40: -97.3252945, 47.7326355, -97.2340546, 47.6662216, -144.9915161, 144.9666748
41: -67.4661102, 40.3551216, -67.3485184, 40.3190918, -107.7852020, 107.7036438
42: -50.0523376, 45.4968834, -49.8779259, 45.4670372, -95.5193634, 95.3748093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1688
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
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1670
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
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1695
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
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1361
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
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
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
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3432837, upper bound: 57.2949341
time: 125.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.4442898, upper bound: 57.4442897
time: 67.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 195.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 195.76
Output dim: 2, lower bound: -57.3432837, upper bound: 57.2949341
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 195.76
Output dim: 2, lower bound: -57.3579696, upper bound: 57.4442936
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 195.76
Output dim: 2, lower bound: -57.3432837, upper bound: 57.2949341
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 195.76
Output dim: 2, lower bound: -57.4442898, upper bound: 57.4442897

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -62.8246002, 46.1394272, -62.6435356, 46.0201111, -108.8447113, 108.7829514
1: -39.9535828, 41.9108429, -39.8411636, 41.8562584, -81.8098450, 81.7520065
2: -37.1589737, 43.9719391, -36.9967575, 43.8576889, -81.0166626, 80.9686890
3: -45.1578178, 52.1529922, -44.9573135, 52.0291023, -97.1869202, 97.1103058
4: -52.7413483, 40.6352463, -52.5871964, 40.5407295, -93.2820740, 93.2224350
5: -46.9859734, 57.1883392, -46.7946815, 57.0425606, -104.0285339, 103.9830170
6: -67.8824005, 41.7486992, -67.8012543, 41.7265167, -109.6089172, 109.5499420
7: -57.2565651, 53.0948524, -57.0679932, 53.0354652, -110.2920303, 110.1628418
8: -47.3799973, 47.2500420, -47.2474327, 47.1503448, -94.5303345, 94.4974747
9: -49.4898643, 52.6811790, -49.3945770, 52.5598564, -102.0497208, 102.0757599
10: -79.2440186, 76.8111725, -78.9900742, 76.4814606, -155.7254791, 155.8012390
11: -80.2167206, 53.0829353, -80.1220398, 52.9466400, -133.1633606, 133.2049713
12: -74.6233978, 58.9171066, -74.3709488, 58.6917419, -133.3151398, 133.2880554
13: -70.9373627, 66.4644394, -70.8559418, 66.4802170, -137.4175720, 137.3203735
14: -106.9116745, 57.2351418, -106.6744537, 57.0781746, -163.9898529, 163.9095764
15: -59.1124039, 50.6385841, -59.0101051, 50.5635147, -109.6759186, 109.6486816
16: -82.9160614, 66.4681702, -82.8232346, 66.3092194, -149.2252808, 149.2914124
17: -119.1143036, 78.7659607, -118.9206696, 78.6453247, -197.7596283, 197.6866150
18: -69.2468185, 42.2520370, -69.1629639, 42.0881195, -111.3349380, 111.4150009
19: -60.1168823, 25.0418930, -60.0300293, 24.9462528, -85.0631332, 85.0719223
20: -54.2249031, 32.4044075, -54.1235275, 32.3465424, -86.5714417, 86.5279388
21: -72.4740448, 36.8109970, -72.3355789, 36.6819763, -109.1560211, 109.1465759
22: -82.0738373, 48.1498756, -82.0288849, 47.9941597, -130.0679932, 130.1787567
23: -54.9241333, 34.8119392, -54.8564301, 34.7111435, -89.6352463, 89.6683655
24: -64.4256134, 34.7492981, -64.4441223, 34.7276535, -99.1532669, 99.1934204
25: -60.1023598, 39.7204475, -60.0990982, 39.6259804, -99.7283325, 99.8195496
26: -92.9245987, 50.8463249, -92.7124252, 50.5545731, -143.4791718, 143.5587463
27: -68.2897491, 44.3648605, -68.2539597, 44.3298187, -112.6195679, 112.6188202
28: -56.6240959, 36.5851517, -56.5667915, 36.5317688, -93.1558533, 93.1519470
29: -81.6157761, 54.3149338, -81.5653229, 54.1554222, -135.7711945, 135.8802490
30: -68.0646896, 37.0546417, -68.0097504, 36.9492264, -105.0139160, 105.0643921
31: -62.7246399, 30.7652683, -62.6466599, 30.6740761, -93.3986969, 93.4119263
32: -65.5979843, 48.0082397, -65.5309525, 47.9946251, -113.5926056, 113.5391922
33: -99.9355011, 58.5091209, -99.8363647, 58.3782120, -158.3137054, 158.3454895
34: -85.1067200, 44.5833893, -84.9969940, 44.4872131, -129.5939331, 129.5803833
35: -80.7843933, 47.4373627, -80.6481857, 47.3309402, -128.1153259, 128.0855408
36: -82.6226120, 48.4857788, -82.4930115, 48.4445343, -131.0671387, 130.9787903
37: -115.4003601, 48.1726494, -115.3496475, 48.0774803, -163.4778290, 163.5222931
38: -102.2234497, 63.6330605, -102.0517883, 63.5928993, -165.8163452, 165.6848450
39: -122.4639282, 54.8366776, -122.3794250, 54.7612267, -177.2251282, 177.2160950
40: -96.7843552, 47.5767288, -96.7013474, 47.5135880, -144.2979431, 144.2780762
41: -67.1182938, 40.0161972, -67.0529709, 39.9662247, -107.0845184, 107.0691605
42: -49.7205963, 44.8140335, -49.6613121, 44.6894684, -94.4100647, 94.4753418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1400
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
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2204525
time: 78.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2886335
time: 91.75 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -62.9991646, 46.1778564, -63.2313194, 46.2420959, -109.2412567, 109.4091797
1: -40.0591087, 41.9433594, -40.1906204, 41.9946175, -82.0537262, 82.1339798
2: -37.3406143, 43.9962997, -37.5958786, 44.0335999, -81.3742065, 81.5921707
3: -45.3461380, 52.1919632, -45.5759659, 52.2583237, -97.6044617, 97.7679214
4: -52.9444046, 40.6714706, -53.2579918, 40.7273788, -93.6717834, 93.9294586
5: -47.1717529, 57.2233124, -47.4108543, 57.2891579, -104.4609070, 104.6341705
6: -67.9457626, 41.8543854, -68.0347595, 42.0747833, -110.0205460, 109.8891449
7: -57.4097519, 53.1290169, -57.5900459, 53.1957397, -110.6054840, 110.7190628
8: -47.5600433, 47.2847595, -47.8429718, 47.3415222, -94.9015656, 95.1277313
9: -49.5501480, 52.8339043, -49.6368675, 53.0717773, -102.6219254, 102.4707718
10: -79.3229218, 77.1613770, -79.4373932, 77.6256638, -156.9485779, 156.5987701
11: -80.2755966, 53.3413582, -80.3827972, 53.7793770, -134.0549774, 133.7241516
12: -74.6666870, 59.2906418, -74.7448883, 59.9225731, -134.5892487, 134.0355225
13: -70.9978180, 66.5592957, -71.0611115, 66.8318253, -137.8296356, 137.6204071
14: -107.0062027, 57.4651909, -107.1465378, 57.8435974, -164.8497925, 164.6117249
15: -59.2696724, 50.7008171, -59.5376663, 50.7960129, -110.0656891, 110.2384796
16: -83.0014038, 66.6567993, -83.1445541, 66.9405365, -149.9419403, 149.8013611
17: -119.1772003, 79.0914764, -119.2819519, 79.7170334, -198.8942108, 198.3734283
18: -69.3212891, 42.3726883, -69.4868011, 42.4907722, -111.8120499, 111.8594894
19: -60.1678047, 25.1248817, -60.2587357, 25.2157078, -85.3835144, 85.3836136
20: -54.2779007, 32.4892502, -54.3590851, 32.6260834, -86.9039841, 86.8483353
21: -72.5285339, 36.9496574, -72.6213226, 37.1405334, -109.6690674, 109.5709839
22: -82.1452332, 48.2713127, -82.3013153, 48.3997307, -130.5449677, 130.5726318
23: -54.9714127, 34.8965073, -55.0541954, 34.9926910, -89.9641037, 89.9506989
24: -64.5125732, 34.7820129, -64.7469330, 34.8292084, -99.3417816, 99.5289307
25: -60.1488647, 39.7944183, -60.2758026, 39.8756142, -100.0244751, 100.0702133
26: -92.9866409, 51.0959091, -93.0969086, 51.3818512, -144.3684998, 144.1928101
27: -68.4102707, 44.3959618, -68.6706619, 44.4401588, -112.8504333, 113.0666199
28: -56.6721420, 36.6263275, -56.7562103, 36.6756897, -93.3478317, 93.3825378
29: -81.6722107, 54.4752579, -81.7821198, 54.6748276, -136.3470306, 136.2573853
30: -68.1147614, 37.1824265, -68.2089996, 37.3752937, -105.4900513, 105.3914261
31: -62.8106346, 30.8288670, -62.9891472, 30.8807449, -93.6913757, 93.8180084
32: -65.6601562, 48.1241379, -65.7495651, 48.3777924, -114.0379486, 113.8737030
33: -100.1156158, 58.5655060, -100.4366684, 58.6533279, -158.7689514, 159.0021667
34: -85.2246552, 44.6329880, -85.3943253, 44.7045593, -129.9292145, 130.0273132
35: -80.9507675, 47.4823570, -81.1915436, 47.5498619, -128.5006256, 128.6738892
36: -82.7185669, 48.5284004, -82.8197479, 48.6112595, -131.3298340, 131.3481445
37: -115.4999084, 48.2376175, -115.6976776, 48.3129082, -163.8128204, 163.9353027
38: -102.3611984, 63.6897316, -102.5167618, 63.8062515, -166.1674500, 166.2064972
39: -122.5981369, 54.8782578, -122.8401184, 54.9365501, -177.5346680, 177.7183838
40: -96.9144287, 47.6042366, -97.1521835, 47.6515732, -144.5659790, 144.7564240
41: -67.1893921, 40.0986176, -67.3056030, 40.2484627, -107.4378510, 107.4042206
42: -49.7720528, 45.0169067, -49.8481560, 45.3532410, -95.1252899, 94.8650665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 821
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
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.3605899
time: 132.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.4401945
time: 77.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.2113876, 46.3707161, -62.7056427, 46.0371170, -109.2484894, 109.0763550
1: -40.1715393, 42.0230293, -39.8736763, 41.8689690, -82.0405121, 81.8967056
2: -37.5300751, 44.2103271, -37.0652122, 43.8674545, -81.3975220, 81.2755280
3: -45.5029526, 52.3703918, -45.0192184, 52.0468483, -97.5498047, 97.3896103
4: -53.2032547, 40.9236183, -52.6708946, 40.5551720, -93.7584229, 93.5945129
5: -47.3417892, 57.4927559, -46.8599052, 57.0600815, -104.4018707, 104.3526382
6: -68.1749725, 42.0699158, -67.8247833, 41.7775993, -109.9525757, 109.8946915
7: -57.5681686, 53.2239571, -57.1144295, 53.0500107, -110.6181564, 110.3383865
8: -47.7998238, 47.5184593, -47.3229828, 47.1648445, -94.9646606, 94.8414383
9: -49.7183189, 53.0549583, -49.4170532, 52.6232796, -102.3415985, 102.4720154
10: -79.7251587, 77.4937897, -79.0202789, 76.6079559, -156.3330994, 156.5140686
11: -80.6382751, 53.6875496, -80.1503601, 53.0630341, -133.7013092, 133.8379059
12: -75.2125320, 59.7862778, -74.3914032, 58.8610802, -134.0736084, 134.1776733
13: -71.1471863, 66.9075775, -70.8759460, 66.5492554, -137.6964417, 137.7835083
14: -107.3780518, 57.7658577, -106.7102432, 57.1835632, -164.5616150, 164.4761047
15: -59.5108795, 50.9715157, -59.0647964, 50.5892639, -110.1001434, 110.0363159
16: -83.2936325, 66.8794022, -82.8614502, 66.3764801, -149.6700897, 149.7408447
17: -119.6341248, 79.6187515, -118.9470444, 78.8126526, -198.4467773, 198.5657959
18: -69.5763550, 42.4653206, -69.1944504, 42.1181107, -111.6944580, 111.6597748
19: -60.3641968, 25.1938438, -60.0525322, 24.9709015, -85.3350983, 85.2463760
20: -54.4757996, 32.6068916, -54.1453514, 32.3823853, -86.8581848, 86.7522354
21: -72.8170547, 37.0906563, -72.3599548, 36.7331734, -109.5502319, 109.4506073
22: -82.3194275, 48.4297409, -82.0532227, 48.0314522, -130.3508606, 130.4829712
23: -55.1374207, 34.9796791, -54.8782349, 34.7370834, -89.8745041, 89.8579102
24: -64.7993317, 34.9365883, -64.5052185, 34.7394028, -99.5387344, 99.4418030
25: -60.3194580, 39.9054565, -60.1252747, 39.6483192, -99.9677734, 100.0307312
26: -93.2355194, 51.2592392, -92.7408752, 50.6251144, -143.8605957, 144.0001221
27: -68.7124023, 44.5027428, -68.3222046, 44.3409615, -113.0533600, 112.8249435
28: -56.8108673, 36.6890907, -56.5874557, 36.5448189, -93.3556824, 93.2765427
29: -81.8269348, 54.6196899, -81.5839691, 54.2076950, -136.0346222, 136.2036591
30: -68.2879791, 37.3590088, -68.0337219, 37.0004730, -105.2884521, 105.3927307
31: -63.0966911, 30.8739033, -62.6939240, 30.6881237, -93.7848129, 93.5678253
32: -65.9335480, 48.3573952, -65.5546722, 48.0600319, -113.9935760, 113.9120560
33: -100.4381790, 58.7845116, -99.9214783, 58.4004936, -158.8386688, 158.7059937
34: -85.3864899, 44.7817459, -85.0420532, 44.5048256, -129.8913116, 129.8237915
35: -81.1654816, 47.6534157, -80.7123718, 47.3488083, -128.5142822, 128.3657837
36: -82.8341522, 48.6403999, -82.5166626, 48.4644279, -131.2985535, 131.1570587
37: -115.7656555, 48.3409653, -115.3982315, 48.0983200, -163.8639832, 163.7391815
38: -102.5408325, 63.8496361, -102.0905914, 63.6227760, -166.1636047, 165.9402313
39: -122.9199066, 55.0085678, -122.4458923, 54.7752838, -177.6951904, 177.4544678
40: -97.1929169, 47.7044487, -96.7628174, 47.5223503, -144.7152710, 144.4672546
41: -67.3940201, 40.2675095, -67.0837555, 40.0080338, -107.4020538, 107.3512650
42: -50.0005989, 45.2913361, -49.6816254, 44.7785873, -94.7791901, 94.9729614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 983
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
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 703
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2204525
time: 75.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2886335
time: 74.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.3859482, 46.4086571, -63.2935982, 46.2588463, -109.6447906, 109.7022552
1: -40.2768478, 42.0556717, -40.2230530, 42.0071907, -82.2840424, 82.2787247
2: -37.7116165, 44.2344398, -37.6643181, 44.0432434, -81.7548599, 81.8987579
3: -45.6913795, 52.4090004, -45.6379852, 52.2757988, -97.9671631, 98.0469818
4: -53.4061661, 40.9596596, -53.3416862, 40.7417603, -94.1479187, 94.3013458
5: -47.5275459, 57.5274315, -47.4761543, 57.3064728, -104.8340073, 105.0035858
6: -68.2379150, 42.1778336, -68.0581818, 42.1261330, -110.3640442, 110.2359924
7: -57.7217903, 53.2580757, -57.6364594, 53.2100677, -110.9318542, 110.8945312
8: -47.9798393, 47.5528831, -47.9186020, 47.3559036, -95.3357391, 95.4714813
9: -49.7778702, 53.2077484, -49.6589470, 53.1353340, -102.9132004, 102.8666992
10: -79.8032227, 77.8437958, -79.4671783, 77.7521286, -157.5553589, 157.3109741
11: -80.6960602, 53.9458771, -80.4108124, 53.8958168, -134.5918732, 134.3566895
12: -75.2554474, 60.1598282, -74.7651672, 60.0920334, -135.3474731, 134.9249878
13: -71.2071228, 67.0032120, -71.0809631, 66.9009857, -138.1081085, 138.0841675
14: -107.4720154, 57.9963341, -107.1820068, 57.9494095, -165.4214172, 165.1783295
15: -59.6745529, 51.0333214, -59.5982094, 50.8216095, -110.4961548, 110.6315231
16: -83.3783112, 67.0712585, -83.1821442, 67.0118103, -150.3901215, 150.2534027
17: -119.6966629, 79.9447632, -119.3081284, 79.8847580, -199.5814209, 199.2528992
18: -69.6514740, 42.5868416, -69.5181732, 42.5214844, -112.1729355, 112.1050110
19: -60.4152832, 25.2766438, -60.2810059, 25.2403603, -85.6556396, 85.5576477
20: -54.5283546, 32.6917496, -54.3806114, 32.6619682, -87.1903076, 87.0723572
21: -72.8710861, 37.2291718, -72.6454468, 37.1916199, -110.0627060, 109.8746185
22: -82.3910828, 48.5503998, -82.3253937, 48.4369316, -130.8280029, 130.8757935
23: -55.1844864, 35.0641022, -55.0758705, 35.0185776, -90.2030640, 90.1399689
24: -64.8868103, 34.9688263, -64.8087692, 34.8408966, -99.7276917, 99.7775955
25: -60.3660583, 39.9788017, -60.3019562, 39.8979225, -100.2639694, 100.2807617
26: -93.2976761, 51.5088730, -93.1251373, 51.4522896, -144.7499695, 144.6340027
27: -68.8334351, 44.5334358, -68.7394028, 44.4511909, -113.2846146, 113.2728424
28: -56.8590584, 36.7301979, -56.7769089, 36.6887627, -93.5478210, 93.5071106
29: -81.8835602, 54.7797165, -81.8010788, 54.7271271, -136.6106873, 136.5807953
30: -68.3380890, 37.4869385, -68.2328262, 37.4266434, -105.7647324, 105.7197647
31: -63.1839638, 30.9373226, -63.0367546, 30.8947182, -94.0786743, 93.9740753
32: -65.9946747, 48.4733849, -65.7731018, 48.4433403, -114.4380188, 114.2464828
33: -100.6185150, 58.8406219, -100.5219498, 58.6751442, -159.2936554, 159.3625641
34: -85.5047913, 44.8311310, -85.4395828, 44.7219086, -130.2266846, 130.2707214
35: -81.3320084, 47.6979027, -81.2557983, 47.5674210, -128.8994293, 128.9537048
36: -82.9301147, 48.6831360, -82.8434448, 48.6308937, -131.5610046, 131.5265808
37: -115.8653030, 48.4057465, -115.7464905, 48.3335342, -164.1988373, 164.1522369
38: -102.6789169, 63.9068298, -102.5558548, 63.8363190, -166.5152283, 166.4626770
39: -123.0549164, 55.0501175, -122.9068298, 54.9504280, -178.0053406, 177.9569397
40: -97.3232269, 47.7320328, -97.2138824, 47.6600151, -144.9832153, 144.9459229
41: -67.4648743, 40.3523903, -67.3362656, 40.2909927, -107.7558594, 107.6886597
42: -50.0513649, 45.4943695, -49.8683815, 45.4424667, -95.4938354, 95.3627472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3381900, upper bound: 57.3605899
time: 536.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.4401947, upper bound: 57.4401944
time: 107.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 646.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2204525
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2886335
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.3605899
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.4401945
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2204525
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.2886335
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.3381900, upper bound: 57.3605899
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 646.96
Output dim: 2, lower bound: -57.4401947, upper bound: 57.4401944

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.4145279, 45.8316803, -62.6049004, 45.9521942, -108.3667221, 108.4365845
1: -39.6814232, 41.7094879, -39.8190308, 41.8244476, -81.5058594, 81.5285110
2: -36.7867966, 43.6962509, -36.9733849, 43.7926559, -80.5794525, 80.6696320
3: -44.7419586, 51.7723770, -44.9315109, 51.9446526, -96.6866074, 96.7038879
4: -52.4366035, 40.4626083, -52.5615463, 40.5110588, -92.9476547, 93.0241470
5: -46.5436058, 56.7611160, -46.7662010, 56.9428596, -103.4864655, 103.5273132
6: -67.6256104, 41.3800621, -67.7707062, 41.6612053, -109.2868118, 109.1507645
7: -56.7854042, 52.7023315, -57.0323830, 52.9510345, -109.7364273, 109.7347107
8: -47.0212479, 46.9483376, -47.2265778, 47.0889244, -94.1101532, 94.1749115
9: -49.1180267, 52.2596283, -49.3178024, 52.5367393, -101.6547699, 101.5774307
10: -78.6461868, 76.0739059, -78.8587112, 76.4446411, -155.0908203, 154.9326172
11: -80.0120316, 52.6799545, -80.0721893, 52.9217567, -132.9337769, 132.7521362
12: -73.8702469, 58.0227814, -74.1868439, 58.6604080, -132.5306549, 132.2096252
13: -70.5583954, 66.0192871, -70.7809448, 66.4444733, -137.0028687, 136.8002167
14: -106.0741577, 56.5229378, -106.4936295, 57.0619507, -163.1361084, 163.0165710
15: -58.6689339, 50.3466415, -58.9339676, 50.5315132, -109.2004471, 109.2806091
16: -82.6201172, 66.0775375, -82.7697220, 66.2684784, -148.8885956, 148.8472595
17: -118.4233704, 77.9514389, -118.7628326, 78.6183167, -197.0416565, 196.7142639
18: -68.8802490, 41.9880333, -69.1040421, 42.0592155, -110.9394379, 111.0920715
19: -59.8834953, 24.9073544, -59.9899635, 24.9323311, -84.8158188, 84.8973160
20: -53.9781418, 32.2241974, -54.0807762, 32.3321609, -86.3103027, 86.3049698
21: -72.1957245, 36.5292320, -72.2846527, 36.6629944, -108.8587189, 108.8138809
22: -81.5176239, 47.7367325, -81.9089890, 47.9690857, -129.4866943, 129.6457214
23: -54.6670647, 34.6684952, -54.8197632, 34.6949768, -89.3620300, 89.4882584
24: -64.1211090, 34.6034317, -64.4137268, 34.7035599, -98.8246689, 99.0171585
25: -59.8366623, 39.4691277, -60.0524216, 39.5996933, -99.4363556, 99.5215454
26: -92.1380463, 50.0994682, -92.5291672, 50.5229187, -142.6609497, 142.6286316
27: -67.8960724, 44.1996384, -68.2210388, 44.2995033, -112.1955719, 112.4206772
28: -56.3751183, 36.4726448, -56.5348129, 36.5123596, -92.8874817, 93.0074615
29: -81.2068787, 53.8440857, -81.4794769, 54.1355286, -135.3424072, 135.3235626
30: -67.7786102, 36.7995911, -67.9761734, 36.9270287, -104.7056274, 104.7757568
31: -62.4068222, 30.6036510, -62.6075134, 30.6521645, -93.0589752, 93.2111664
32: -65.3750916, 47.7507172, -65.4973145, 47.9713860, -113.3464661, 113.2480316
33: -99.4807358, 58.3298187, -99.8011017, 58.3405228, -157.8212585, 158.1309204
34: -84.7020569, 44.3526726, -84.9655609, 44.4419746, -129.1440277, 129.3182220
35: -80.3743896, 47.2494774, -80.6180725, 47.2951355, -127.6695251, 127.8675537
36: -82.3196869, 48.3397751, -82.4579315, 48.4171600, -130.7368469, 130.7976990
37: -115.0157242, 47.9670792, -115.3035660, 48.0470543, -163.0627747, 163.2706451
38: -101.7705307, 63.3715439, -102.0161972, 63.5490837, -165.3195801, 165.3877258
39: -122.0890732, 54.5915756, -122.3394394, 54.7218742, -176.8109436, 176.9310150
40: -96.3511581, 47.2769470, -96.6709824, 47.4418793, -143.7930298, 143.9479218
41: -66.8181915, 39.7359772, -67.0275421, 39.9184990, -106.7366943, 106.7635193
42: -49.5324249, 44.4778252, -49.6307449, 44.6543198, -94.1867371, 94.1085663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 677
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1767
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
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1515
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
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.1464764
time: 84.10 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2114688
time: 83.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -62.7904701, 46.1111069, -62.6349602, 46.0131493, -108.8036194, 108.7460632
1: -39.9348679, 41.8854599, -39.8365021, 41.8498383, -81.7846985, 81.7219620
2: -37.1322784, 43.9523239, -36.9902496, 43.8529358, -80.9852142, 80.9425735
3: -45.1327362, 52.0918465, -44.9511528, 52.0136757, -97.1464081, 97.0429840
4: -52.7074242, 40.6183739, -52.5789146, 40.5365486, -93.2439728, 93.1972885
5: -46.9635811, 57.1269913, -46.7890816, 57.0279350, -103.9915161, 103.9160767
6: -67.8618927, 41.6507568, -67.7961502, 41.6994629, -109.5613480, 109.4469070
7: -57.2332726, 53.0249939, -57.0622253, 53.0165215, -110.2497864, 110.0872192
8: -47.3491592, 47.2316780, -47.2398987, 47.1457787, -94.4949341, 94.4715729
9: -49.4613609, 52.6601486, -49.3874435, 52.5547256, -102.0160828, 102.0475922
10: -79.2105408, 76.7667999, -78.9818344, 76.4704590, -155.6809692, 155.7486267
11: -80.1901398, 53.0383224, -80.1155396, 52.9358139, -133.1259460, 133.1538544
12: -74.5835419, 58.8726768, -74.3612366, 58.6808777, -133.2644043, 133.2339172
13: -70.9026871, 66.4302979, -70.8471832, 66.4719849, -137.3746643, 137.2774658
14: -106.8568497, 57.2197037, -106.6606827, 57.0744019, -163.9312439, 163.8803711
15: -58.9839783, 50.6194458, -58.9779358, 50.5587883, -109.5427704, 109.5973816
16: -82.8882294, 66.4312668, -82.8161545, 66.3002014, -149.1884308, 149.2474213
17: -119.0696716, 78.6987762, -118.9095917, 78.6292725, -197.6989136, 197.6083374
18: -69.2113953, 42.2254295, -69.1541901, 42.0815086, -111.2929077, 111.3796234
19: -60.0957718, 25.0251484, -60.0245934, 24.9421730, -85.0379486, 85.0497437
20: -54.2048264, 32.3891907, -54.1184769, 32.3427811, -86.5476074, 86.5076599
21: -72.4467850, 36.7950211, -72.3285828, 36.6780090, -109.1247864, 109.1236038
22: -81.9610443, 48.1296921, -82.0022278, 47.9892807, -129.9503174, 130.1319275
23: -54.9073868, 34.7946892, -54.8521423, 34.7068787, -89.6142578, 89.6468353
24: -64.3998337, 34.7342873, -64.4378510, 34.7237701, -99.1236038, 99.1721344
25: -60.0488472, 39.7048035, -60.0863342, 39.6221237, -99.6709747, 99.7911377
26: -92.8334198, 50.8224525, -92.6875687, 50.5485916, -143.3820190, 143.5100250
27: -68.2547455, 44.3429031, -68.2453995, 44.3243294, -112.5790710, 112.5882950
28: -56.6079674, 36.5703812, -56.5625648, 36.5281258, -93.1360779, 93.1329498
29: -81.5702515, 54.2962952, -81.5529709, 54.1509399, -135.7211761, 135.8492584
30: -68.0412292, 37.0272751, -68.0039902, 36.9425125, -104.9837341, 105.0312653
31: -62.6982079, 30.7460632, -62.6400414, 30.6693459, -93.3675537, 93.3861084
32: -65.5757446, 47.9812965, -65.5254059, 47.9880943, -113.5638351, 113.5066986
33: -99.9024124, 58.4866295, -99.8282242, 58.3725662, -158.2749786, 158.3148499
34: -85.0820160, 44.5581436, -84.9908905, 44.4806824, -129.5626984, 129.5490112
35: -80.7498093, 47.4184799, -80.6396561, 47.3262253, -128.0760345, 128.0581360
36: -82.5970993, 48.4706650, -82.4866562, 48.4407730, -131.0378723, 130.9573212
37: -115.3586578, 48.1543274, -115.3395081, 48.0729599, -163.4316101, 163.4938354
38: -102.1942368, 63.6084366, -102.0445862, 63.5867844, -165.7810211, 165.6530151
39: -122.4108124, 54.8176613, -122.3661652, 54.7562408, -177.1670532, 177.1838226
40: -96.7567902, 47.5508881, -96.6945877, 47.5067215, -144.2634888, 144.2454834
41: -67.1001740, 39.9637184, -67.0484695, 39.9512329, -107.0514069, 107.0121918
42: -49.7015381, 44.7677803, -49.6566162, 44.6781273, -94.3796616, 94.4243927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1767
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
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 871
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
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
time: 77.08 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3345721, upper bound: 57.2797078
time: 83.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5883408, 45.8703575, -63.1928215, 46.1743164, -108.7626572, 109.0631790
1: -39.7865372, 41.7420120, -40.1686134, 41.9627953, -81.7493286, 81.9106293
2: -36.9677238, 43.7207489, -37.5725365, 43.9699478, -80.9376678, 81.2932816
3: -44.9290810, 51.8113441, -45.5502052, 52.1744118, -97.1034927, 97.3615417
4: -52.6390190, 40.4986954, -53.2325859, 40.6975937, -93.3366089, 93.7312698
5: -46.7316933, 56.7960510, -47.3824120, 57.1914253, -103.9231110, 104.1784668
6: -67.6885986, 41.4848099, -68.0041046, 42.0094643, -109.6980591, 109.4889145
7: -56.9364777, 52.7363510, -57.5545349, 53.1113586, -110.0478287, 110.2908859
8: -47.2009048, 46.9831314, -47.8221550, 47.2801933, -94.4810944, 94.8052750
9: -49.1783867, 52.4118233, -49.5601730, 53.0487366, -102.2271271, 101.9719925
10: -78.7253723, 76.4231415, -79.3066101, 77.5891876, -156.3145599, 155.7297363
11: -80.0694275, 52.9373131, -80.3326263, 53.7544556, -133.8238831, 133.2699280
12: -73.9138031, 58.3951111, -74.5610046, 59.8913612, -133.8051605, 132.9561157
13: -70.6189575, 66.1131363, -70.9861298, 66.7957001, -137.4146576, 137.0992737
14: -106.1690369, 56.7524567, -106.9654388, 57.8273964, -163.9964142, 163.7178955
15: -58.8274422, 50.4082184, -59.4661179, 50.7639313, -109.5913696, 109.8743286
16: -82.7044144, 66.2639542, -83.0907059, 66.9005127, -149.6049194, 149.3546600
17: -118.4864960, 78.2756958, -119.1241074, 79.6900330, -198.1765289, 197.3998108
18: -68.9542923, 42.1082001, -69.4279785, 42.4618797, -111.4161682, 111.5361786
19: -59.9343338, 24.9915371, -60.2185631, 25.2017784, -85.1361084, 85.2100983
20: -54.0312157, 32.3085251, -54.3158035, 32.6117706, -86.6429825, 86.6243210
21: -72.2501831, 36.6674004, -72.5699844, 37.1214867, -109.3716583, 109.2373810
22: -81.5892029, 47.8579254, -82.1824417, 48.3746872, -129.9638824, 130.0403748
23: -54.7137566, 34.7527695, -55.0172920, 34.9766235, -89.6903763, 89.7700653
24: -64.2070160, 34.6358261, -64.7171326, 34.8051071, -99.0121078, 99.3529510
25: -59.8829231, 39.5426140, -60.2290840, 39.8493958, -99.7323151, 99.7716980
26: -92.2005539, 50.3472137, -92.9135742, 51.3503914, -143.5509491, 143.2607880
27: -68.0157700, 44.2306633, -68.6378708, 44.4102440, -112.4260101, 112.8685303
28: -56.4226265, 36.5137634, -56.7241936, 36.6563873, -93.0790100, 93.2379608
29: -81.2635727, 54.0036545, -81.6963501, 54.6550140, -135.9185791, 135.7000122
30: -67.8276367, 36.9269981, -68.1752396, 37.3531265, -105.1807556, 105.1022339
31: -62.4901047, 30.6669712, -62.9499245, 30.8588619, -93.3489685, 93.6168976
32: -65.4368439, 47.8657532, -65.7157593, 48.3545151, -113.7913589, 113.5815125
33: -99.6600189, 58.3862152, -100.4013901, 58.6150818, -158.2750854, 158.7875977
34: -84.8191681, 44.4023857, -85.3629456, 44.6590576, -129.4782257, 129.7653351
35: -80.5397797, 47.2948380, -81.1615295, 47.5135727, -128.0533295, 128.4563599
36: -82.4137192, 48.3811874, -82.7846680, 48.5836220, -130.9973297, 131.1658630
37: -115.1147003, 48.0317917, -115.6517334, 48.2822342, -163.3969421, 163.6835327
38: -101.9073105, 63.4278297, -102.4812622, 63.7627869, -165.6701050, 165.9090881
39: -122.2233505, 54.6328735, -122.8003387, 54.8969498, -177.1203003, 177.4332123
40: -96.4807587, 47.3043442, -97.1219635, 47.5796318, -144.0603943, 144.4263000
41: -66.8883896, 39.8209152, -67.2801056, 40.2023087, -107.0906982, 107.1010208
42: -49.5831947, 44.6786766, -49.8175049, 45.3184967, -94.9016876, 94.4961853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 902
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
type: B, layer: 1, pos: 1679
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
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 629
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
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1403
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
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1569
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

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2685506
time: 71.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2114688
time: 75.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -62.9652443, 46.1495285, -63.2228889, 46.2352219, -109.2004547, 109.3724213
1: -40.0404739, 41.9181747, -40.1859741, 41.9883690, -82.0288391, 82.1041489
2: -37.3141022, 43.9767303, -37.5895233, 44.0288811, -81.3429718, 81.5662537
3: -45.3211670, 52.1306877, -45.5699310, 52.2420502, -97.5632172, 97.7006149
4: -52.9107666, 40.6547050, -53.2499046, 40.7232208, -93.6339874, 93.9046021
5: -47.1494827, 57.1619110, -47.4053345, 57.2726631, -104.4221497, 104.5672379
6: -67.9253540, 41.7572250, -68.0297089, 42.0493889, -109.9747467, 109.7869263
7: -57.3864784, 53.0586548, -57.5843468, 53.1768112, -110.5632935, 110.6429977
8: -47.5293732, 47.2664070, -47.8356361, 47.3369598, -94.8663254, 95.1020432
9: -49.5218048, 52.8129501, -49.6299362, 53.0667572, -102.5885544, 102.4428864
10: -79.2894211, 77.1173096, -79.4291687, 77.6147232, -156.9041443, 156.5464783
11: -80.2491302, 53.2971573, -80.3763885, 53.7690010, -134.0181274, 133.6735535
12: -74.6268997, 59.2465553, -74.7352448, 59.9120865, -134.5389862, 133.9817810
13: -70.9634323, 66.5251541, -71.0524979, 66.8236694, -137.7870789, 137.5776520
14: -106.9515228, 57.4497643, -107.1330795, 57.8397789, -164.7912903, 164.5828247
15: -59.1414719, 50.6817436, -59.5051537, 50.7913094, -109.9327850, 110.1868973
16: -82.9736328, 66.6201782, -83.1375656, 66.9318390, -149.9054718, 149.7577362
17: -119.1327820, 79.0248260, -119.2711411, 79.7014313, -198.8342133, 198.2959595
18: -69.2864532, 42.3458633, -69.4783936, 42.4841728, -111.7706146, 111.8242569
19: -60.1468964, 25.1082668, -60.2534103, 25.2117329, -85.3586273, 85.3616791
20: -54.2579002, 32.4741058, -54.3541031, 32.6223907, -86.8802948, 86.8282089
21: -72.5013123, 36.9337692, -72.6144257, 37.1366348, -109.6379395, 109.5481873
22: -82.0327301, 48.2511024, -82.2748642, 48.3948479, -130.4275818, 130.5259705
23: -54.9546928, 34.8793869, -55.0499306, 34.9884949, -89.9431915, 89.9293060
24: -64.4869156, 34.7670059, -64.7406769, 34.8253441, -99.3122482, 99.5076752
25: -60.0955811, 39.7787819, -60.2631798, 39.8717499, -99.9673309, 100.0419617
26: -92.8953857, 51.0720711, -93.0722275, 51.3758469, -144.2711945, 144.1442871
27: -68.3753662, 44.3740883, -68.6621399, 44.4346886, -112.8100433, 113.0362244
28: -56.6560516, 36.6115417, -56.7519951, 36.6721115, -93.3281631, 93.3635406
29: -81.6266937, 54.4566116, -81.7698288, 54.6703415, -136.2970276, 136.2264404
30: -68.0913239, 37.1552887, -68.2033386, 37.3687820, -105.4600906, 105.3586273
31: -62.7842598, 30.8097572, -62.9825821, 30.8760338, -93.6602936, 93.7923279
32: -65.6379929, 48.0973282, -65.7440491, 48.3714638, -114.0094604, 113.8413773
33: -100.0826797, 58.5430984, -100.4287033, 58.6477013, -158.7303772, 158.9718018
34: -85.2000046, 44.6077881, -85.3882141, 44.6981163, -129.8981171, 129.9959869
35: -80.9164047, 47.4634895, -81.1831741, 47.5452385, -128.4616394, 128.6466522
36: -82.6931458, 48.5132561, -82.8134308, 48.6075974, -131.3007202, 131.3266907
37: -115.4586563, 48.2192764, -115.6878586, 48.3084068, -163.7670593, 163.9071350
38: -102.3320770, 63.6652603, -102.5096130, 63.8003006, -166.1323853, 166.1748657
39: -122.5457306, 54.8592911, -122.8273392, 54.9315910, -177.4772949, 177.6866150
40: -96.8869858, 47.5785141, -97.1455307, 47.6448059, -144.5317993, 144.7240448
41: -67.1713333, 40.0460587, -67.3011627, 40.2331772, -107.4045105, 107.3472214
42: -49.7530518, 44.9710655, -49.8434830, 45.3422737, -95.0953217, 94.8145447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 680

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3489547
time: 87.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493
time: 69.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -62.8046951, 46.0652275, -62.6670532, 45.9692001, -108.7738876, 108.7322845
1: -39.9005623, 41.8218460, -39.8515472, 41.8371315, -81.7376938, 81.6733932
2: -37.1589966, 43.9354401, -37.0419083, 43.8024368, -80.9614258, 80.9773483
3: -45.0948486, 51.9910622, -44.9934616, 51.9624825, -97.0573273, 96.9845123
4: -52.8999977, 40.7519913, -52.6454086, 40.5254669, -93.4254608, 93.3973999
5: -46.9056854, 57.0705299, -46.8314552, 56.9604301, -103.8661194, 103.9019852
6: -67.9221954, 41.6990967, -67.7942200, 41.7125549, -109.6347504, 109.4933090
7: -57.1028442, 52.8300323, -57.0787773, 52.9652634, -110.0681076, 109.9088058
8: -47.4428139, 47.2181053, -47.3022461, 47.1034889, -94.5462952, 94.5203476
9: -49.3482170, 52.6350212, -49.3402939, 52.6002693, -101.9484863, 101.9753113
10: -79.1299286, 76.7587280, -78.8889923, 76.5713654, -155.7012787, 155.6477203
11: -80.4573059, 53.2853661, -80.1004486, 53.0383072, -133.4956055, 133.3858185
12: -74.4621964, 58.8941345, -74.2073669, 58.8298683, -133.2920532, 133.1015015
13: -70.7701340, 66.4650574, -70.8009491, 66.5135651, -137.2836914, 137.2660065
14: -106.5428543, 57.0551529, -106.5294724, 57.1674347, -163.7102814, 163.5846252
15: -59.0897484, 50.6836624, -58.9889297, 50.5572014, -109.6469498, 109.6725922
16: -83.0093689, 66.4824753, -82.8079300, 66.3364868, -149.3458557, 149.2904053
17: -118.9447708, 78.8076248, -118.7892838, 78.7856827, -197.7304535, 197.5968933
18: -69.2117462, 42.2012558, -69.1356277, 42.0891724, -111.3009186, 111.3368835
19: -60.1312065, 25.0607853, -60.0124779, 24.9569798, -85.0881882, 85.0732574
20: -54.2313957, 32.4270554, -54.1025658, 32.3680458, -86.5994339, 86.5296173
21: -72.5401917, 36.8094330, -72.3089752, 36.7142181, -109.2544098, 109.1184082
22: -81.7618027, 48.0204468, -81.9335938, 48.0062561, -129.7680359, 129.9540405
23: -54.8811111, 34.8369408, -54.8415489, 34.7209473, -89.6020508, 89.6784897
24: -64.4961243, 34.7916374, -64.4750214, 34.7152901, -99.2114105, 99.2666626
25: -60.0532341, 39.6552277, -60.0786819, 39.6219940, -99.6752319, 99.7338943
26: -92.4537582, 50.5232773, -92.5579147, 50.5933685, -143.0471191, 143.0811768
27: -68.3208008, 44.3367653, -68.2893524, 44.3106842, -112.6314850, 112.6261139
28: -56.5623055, 36.5774040, -56.5554466, 36.5254135, -93.0877228, 93.1328430
29: -81.4141617, 54.1513214, -81.4981689, 54.1877670, -135.6019287, 135.6494904
30: -68.0019379, 37.1051903, -68.0001373, 36.9783096, -104.9802399, 105.1053162
31: -62.7720985, 30.7133503, -62.6548157, 30.6661797, -93.4382782, 93.3681641
32: -65.7216644, 48.1013031, -65.5209961, 48.0368652, -113.7585297, 113.6222992
33: -99.9845428, 58.6090088, -99.8862839, 58.3627663, -158.3473053, 158.4952850
34: -84.9816589, 44.5521584, -85.0106506, 44.4595451, -129.4411926, 129.5628052
35: -80.7562943, 47.4677200, -80.6823273, 47.3129692, -128.0692596, 128.1500397
36: -82.5314713, 48.4966087, -82.4815598, 48.4370461, -130.9685059, 130.9781494
37: -115.3833389, 48.1368790, -115.3523026, 48.0679245, -163.4512634, 163.4891815
38: -102.0907822, 63.5881996, -102.0550537, 63.5790634, -165.6698303, 165.6432495
39: -122.5488739, 54.7653809, -122.4060135, 54.7359009, -177.2847748, 177.1713867
40: -96.7624435, 47.4041328, -96.7325134, 47.4505844, -144.2130280, 144.1366425
41: -67.0916214, 39.9899445, -67.0583191, 39.9606247, -107.0522461, 107.0482635
42: -49.8154068, 44.9546967, -49.6510239, 44.7436295, -94.5590363, 94.6057205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 677
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.1464764
time: 91.10 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2114688
time: 89.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -63.1770935, 46.3423615, -62.6969872, 46.0301208, -109.2071991, 109.0393524
1: -40.1524620, 41.9979095, -39.8689384, 41.8625603, -82.0150146, 81.8668442
2: -37.5029907, 44.1905403, -37.0586090, 43.8626785, -81.3656616, 81.2491455
3: -45.4776573, 52.3053169, -45.0130157, 52.0312691, -97.5089264, 97.3183289
4: -53.1683807, 40.9066162, -52.6624184, 40.5509491, -93.7193298, 93.5690308
5: -47.3193932, 57.4312286, -46.8542442, 57.0454369, -104.3648300, 104.2854691
6: -68.1542053, 41.9632416, -67.8196640, 41.7503586, -109.9045410, 109.7829056
7: -57.5444641, 53.1539536, -57.1086082, 53.0310402, -110.5754929, 110.2625580
8: -47.7681313, 47.4997444, -47.3153000, 47.1602669, -94.9283981, 94.8150482
9: -49.6899567, 53.0330238, -49.4098778, 52.6180534, -102.3080139, 102.4429016
10: -79.6913147, 77.4478302, -79.0119705, 76.5967102, -156.2880249, 156.4597931
11: -80.6112213, 53.6423836, -80.1437836, 53.0520782, -133.6632996, 133.7861633
12: -75.1722260, 59.7409134, -74.3816757, 58.8499908, -134.0222168, 134.1225891
13: -71.1125183, 66.8720398, -70.8671341, 66.5408020, -137.6533203, 137.7391663
14: -107.3232117, 57.7500038, -106.6964264, 57.1797295, -164.5029449, 164.4464264
15: -59.3676071, 50.9519997, -59.0324898, 50.5845070, -109.9521179, 109.9844894
16: -83.2653580, 66.8413849, -82.8543167, 66.3673248, -149.6326447, 149.6957092
17: -119.5893021, 79.5496674, -118.9359818, 78.7961731, -198.3854675, 198.4856415
18: -69.5399017, 42.4381790, -69.1855164, 42.1114311, -111.6513367, 111.6236877
19: -60.3427811, 25.1770115, -60.0470200, 24.9667912, -85.3095551, 85.2240295
20: -54.4556122, 32.5916061, -54.1402626, 32.3786011, -86.8342056, 86.7318726
21: -72.7895050, 37.0745735, -72.3528748, 36.7291870, -109.5186768, 109.4274445
22: -82.2124710, 48.4092064, -82.0264282, 48.0265160, -130.2389832, 130.4356384
23: -55.1204987, 34.9622154, -54.8739166, 34.7327690, -89.8532715, 89.8361282
24: -64.7725677, 34.9214516, -64.4987640, 34.7355194, -99.5080795, 99.4202118
25: -60.2661209, 39.8895798, -60.1124382, 39.6444130, -99.9105377, 100.0020142
26: -93.1439972, 51.2348213, -92.7158432, 50.6190262, -143.7630310, 143.9506531
27: -68.6758118, 44.4816971, -68.3133850, 44.3354492, -113.0112381, 112.7950821
28: -56.7946472, 36.6741219, -56.5832138, 36.5411758, -93.3358078, 93.2573395
29: -81.7814407, 54.6008873, -81.5715027, 54.2031975, -135.9846344, 136.1723938
30: -68.2642212, 37.3312607, -68.0278854, 36.9936638, -105.2578735, 105.3591385
31: -63.0697746, 30.8547821, -62.6871834, 30.6833572, -93.7531281, 93.5419617
32: -65.9109573, 48.3299446, -65.5490570, 48.0534019, -113.9643555, 113.8789978
33: -100.4044189, 58.7619247, -99.9132538, 58.3947334, -158.7991486, 158.6751709
34: -85.3611755, 44.7563705, -85.0358658, 44.4982071, -129.8593750, 129.7922363
35: -81.1303101, 47.6344261, -80.7037735, 47.3440323, -128.4743347, 128.3381958
36: -82.8082733, 48.6252289, -82.5102692, 48.4606323, -131.2688904, 131.1354828
37: -115.7235641, 48.3227310, -115.3880081, 48.0937424, -163.8173065, 163.7107239
38: -102.5105057, 63.8248901, -102.0832825, 63.6166000, -166.1270752, 165.9081573
39: -122.8647232, 54.9895630, -122.4325180, 54.7702560, -177.6349792, 177.4220581
40: -97.1647644, 47.6784515, -96.7559433, 47.5154610, -144.6802216, 144.4343872
41: -67.3755798, 40.2060471, -67.0792389, 39.9929237, -107.3684845, 107.2852859
42: -49.9812279, 45.2436485, -49.6768837, 44.7669525, -94.7481842, 94.9205322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1688
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
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1670
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
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1713
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
type: B, layer: 1, pos: 1342
type: B, layer: 1, pos: 871
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
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
time: 75.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2797078
time: 72.85 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -62.9785538, 46.1034927, -63.2551384, 46.1910591, -109.1696014, 109.3586273
1: -40.0055313, 41.8544044, -40.2010193, 41.9753723, -81.9808960, 82.0554199
2: -37.3398552, 43.9597206, -37.6410484, 43.9796066, -81.3194580, 81.6007614
3: -45.2820587, 52.0296555, -45.6122398, 52.1919250, -97.4739838, 97.6418839
4: -53.1022491, 40.7878761, -53.3164825, 40.7119751, -93.8142242, 94.1043549
5: -47.0936584, 57.1052017, -47.4477615, 57.2087593, -104.3024139, 104.5529633
6: -67.9848328, 41.8062973, -68.0276184, 42.0609398, -110.0457611, 109.8339157
7: -57.2539406, 52.8641434, -57.6008644, 53.1253395, -110.3792801, 110.4650116
8: -47.6224213, 47.2526474, -47.8978844, 47.2945900, -94.9169922, 95.1505280
9: -49.4078369, 52.7873611, -49.5822945, 53.1124191, -102.5202560, 102.3696518
10: -79.2081833, 77.1077652, -79.3364563, 77.7158356, -156.9240112, 156.4442139
11: -80.5131378, 53.5427017, -80.3605957, 53.8710709, -134.3842163, 133.9032898
12: -74.5053711, 59.2665100, -74.5813370, 60.0608749, -134.5662537, 133.8478394
13: -70.8302612, 66.5598145, -71.0059891, 66.8649063, -137.6951599, 137.5657959
14: -106.6372299, 57.2851601, -107.0009995, 57.9333344, -164.5705566, 164.2861633
15: -59.2549820, 50.7449226, -59.5320892, 50.7894592, -110.0444412, 110.2770081
16: -83.0929947, 66.6729889, -83.1283340, 66.9726334, -150.0656128, 149.8013306
17: -119.0074921, 79.1324158, -119.1503143, 79.8578186, -198.8653107, 198.2827301
18: -69.2864532, 42.3220177, -69.4594574, 42.4925804, -111.7790375, 111.7814789
19: -60.1823158, 25.1448536, -60.2408676, 25.2264118, -85.4087219, 85.3857193
20: -54.2840843, 32.5114365, -54.3372765, 32.6476860, -86.9317703, 86.8487091
21: -72.5942459, 36.9474602, -72.5940552, 37.1726074, -109.7668533, 109.5415115
22: -81.8337097, 48.1409836, -82.2068481, 48.4118423, -130.2455444, 130.3478394
23: -54.9276161, 34.9210739, -55.0389786, 35.0025291, -89.9301453, 89.9600372
24: -64.5825500, 34.8235664, -64.7790833, 34.8167915, -99.3993378, 99.6026459
25: -60.0998230, 39.7282333, -60.2552834, 39.8716888, -99.9715118, 99.9835052
26: -92.5162735, 50.7707787, -92.9419556, 51.4207840, -143.9370575, 143.7127380
27: -68.4410400, 44.3675232, -68.7067032, 44.4213142, -112.8623505, 113.0742264
28: -56.6099663, 36.6184921, -56.7449226, 36.6694336, -93.2793961, 93.3634186
29: -81.4709702, 54.3106461, -81.7154922, 54.7073212, -136.1782837, 136.0261383
30: -68.0509033, 37.2327194, -68.1990356, 37.4045029, -105.4553986, 105.4317551
31: -62.8576927, 30.7765274, -62.9975662, 30.8728218, -93.7305069, 93.7740936
32: -65.7824097, 48.2164192, -65.7392960, 48.4201813, -114.2025757, 113.9557037
33: -100.1640625, 58.6650124, -100.4867477, 58.6368484, -158.8009033, 159.1517639
34: -85.0991287, 44.6016884, -85.4081955, 44.6764221, -129.7755432, 130.0098877
35: -80.9219055, 47.5125618, -81.2258301, 47.5311432, -128.4530487, 128.7383881
36: -82.6266632, 48.5383759, -82.8083191, 48.6032028, -131.2298584, 131.3466949
37: -115.4831772, 48.2013474, -115.7005463, 48.3028946, -163.7860718, 163.9018860
38: -102.2279053, 63.6450386, -102.5203705, 63.7928391, -166.0207214, 166.1654053
39: -122.6838837, 54.8066864, -122.8670959, 54.9108353, -177.5947266, 177.6737823
40: -96.8924026, 47.4316101, -97.1836929, 47.5880432, -144.4804382, 144.6152954
41: -67.1615906, 40.0754242, -67.3107452, 40.2453613, -107.4069519, 107.3861694
42: -49.8653679, 45.1562920, -49.8376923, 45.4079056, -95.2732697, 94.9939728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 902
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
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1400
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
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1695
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
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 629
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1361
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
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2685495
time: 68.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3585403
time: 85.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.3518677, 46.3803329, -63.2850876, 46.2519379, -109.6038055, 109.6654205
1: -40.2578659, 42.0305672, -40.2183342, 42.0009384, -82.2587967, 82.2489014
2: -37.6847343, 44.2146683, -37.6578979, 44.0385017, -81.7232361, 81.8725662
3: -45.6661835, 52.3437996, -45.6318512, 52.2593918, -97.9255753, 97.9756470
4: -53.3715439, 40.9427032, -53.3334007, 40.7375793, -94.1091003, 94.2761002
5: -47.5051804, 57.4658051, -47.4706039, 57.2899895, -104.7951660, 104.9364090
6: -68.2171631, 42.0726738, -68.0530624, 42.1006088, -110.3177719, 110.1257324
7: -57.6980896, 53.1876640, -57.6306953, 53.1910934, -110.8891830, 110.8183594
8: -47.9482651, 47.5342026, -47.9111023, 47.3512955, -95.2995605, 95.4452972
9: -49.7496185, 53.1859245, -49.6519737, 53.1302185, -102.8798370, 102.8378983
10: -79.7693481, 77.7980652, -79.4589233, 77.7409210, -157.5102692, 157.2569885
11: -80.6691132, 53.9011574, -80.4043732, 53.8852539, -134.5543518, 134.3055115
12: -75.2151794, 60.1147461, -74.7555237, 60.0812683, -135.2964478, 134.8702698
13: -71.1726990, 66.9677277, -71.0723038, 66.8926086, -138.0653076, 138.0400238
14: -107.4172897, 57.9804993, -107.1685104, 57.9455414, -165.3628235, 165.1490173
15: -59.5287399, 51.0138550, -59.5626411, 50.8168869, -110.3456268, 110.5764923
16: -83.3500748, 67.0334625, -83.1750870, 67.0029984, -150.3530731, 150.2085419
17: -119.6519852, 79.8761292, -119.2972641, 79.8687820, -199.5207672, 199.1734009
18: -69.6155853, 42.5595856, -69.5096130, 42.5148277, -112.1304016, 112.0691986
19: -60.3940277, 25.2598686, -60.2756157, 25.2363205, -85.6303482, 85.5354843
20: -54.5081978, 32.6765442, -54.3755913, 32.6582832, -87.1664810, 87.0521240
21: -72.8435516, 37.2131500, -72.6384888, 37.1877022, -110.0312500, 109.8516388
22: -82.2841797, 48.5298500, -82.2988586, 48.4320335, -130.7162018, 130.8287048
23: -55.1675797, 35.0467186, -55.0715981, 35.0143127, -90.1818924, 90.1183167
24: -64.8600922, 34.9537201, -64.8023987, 34.8370438, -99.6971283, 99.7561188
25: -60.3128853, 39.9629822, -60.2892838, 39.8940353, -100.2069016, 100.2522659
26: -93.2060394, 51.4844589, -93.1003189, 51.4462280, -144.6522675, 144.5847778
27: -68.7969055, 44.5125008, -68.7306671, 44.4457169, -113.2426071, 113.2431564
28: -56.8428574, 36.7152176, -56.7726860, 36.6851273, -93.5279846, 93.4878998
29: -81.8380508, 54.7608719, -81.7886276, 54.7226524, -136.5606995, 136.5494995
30: -68.3143768, 37.4593811, -68.2270966, 37.4200363, -105.7344131, 105.6864624
31: -63.1571121, 30.9183006, -63.0300713, 30.8900166, -94.0471268, 93.9483566
32: -65.9721222, 48.4460716, -65.7675934, 48.4369125, -114.4090118, 114.2136612
33: -100.5849152, 58.8180161, -100.5139160, 58.6695213, -159.2544250, 159.3319397
34: -85.4794693, 44.8057785, -85.4333954, 44.7154083, -130.1948700, 130.2391663
35: -81.2970657, 47.6788750, -81.2474060, 47.5627594, -128.8598328, 128.9262848
36: -82.9043427, 48.6679306, -82.8370819, 48.6272011, -131.5315399, 131.5050049
37: -115.8236847, 48.3875504, -115.7365723, 48.3290253, -164.1526947, 164.1241150
38: -102.6486511, 63.8821793, -102.5485992, 63.8303452, -166.4790039, 166.4307709
39: -123.0005035, 55.0311546, -122.8939972, 54.9455109, -177.9459991, 177.9251556
40: -97.2952194, 47.7062263, -97.2071762, 47.6532364, -144.9484558, 144.9133911
41: -67.4464874, 40.2906189, -67.3317719, 40.2756042, -107.7220764, 107.6223907
42: -50.0320473, 45.4470367, -49.8636627, 45.4312592, -95.4632950, 95.3106995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1688
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
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1670
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
type: B, layer: 1, pos: 629
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
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
time: 78.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493
time: 80.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 161.32 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.1464764
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2114688
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3345721, upper bound: 57.2797078
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2685506
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2114688
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3489547
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.1464764
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2114688
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2797078
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2685495
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3585403
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 161.32
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -62.2945099, 45.7984619, -62.1588593, 45.8283577, -108.1228638, 107.9573212
1: -39.6128845, 41.6806335, -39.5679550, 41.7157440, -81.3286285, 81.2485886
2: -36.6749039, 43.6731186, -36.5593567, 43.7069397, -80.3818359, 80.2324677
3: -44.6127281, 51.7393227, -44.4497910, 51.8200302, -96.4327545, 96.1891174
4: -52.3021240, 40.4331055, -52.0701904, 40.4013824, -92.7035065, 92.5032959
5: -46.4269867, 56.7305298, -46.3303146, 56.8281403, -103.2551270, 103.0608368
6: -67.5746613, 41.3174133, -67.5801315, 41.4368896, -109.0115433, 108.8975372
7: -56.6964951, 52.6746178, -56.6995659, 52.8475647, -109.5440598, 109.3741837
8: -46.9232559, 46.9150658, -46.8683319, 46.9647827, -93.8880310, 93.7833939
9: -49.0485840, 52.1894798, -49.0621262, 52.2847404, -101.3333130, 101.2516022
10: -78.5856857, 75.8271790, -78.6343689, 75.5346527, -154.1203308, 154.4615326
11: -79.9630280, 52.4539185, -79.8864899, 52.0922241, -132.0552521, 132.3403931
12: -73.8325806, 57.8005066, -74.0463104, 57.8292236, -131.6618042, 131.8468018
13: -70.4477081, 65.9520645, -70.3714066, 66.1911926, -136.6388855, 136.3234558
14: -105.9957199, 56.3437767, -106.1996765, 56.3851166, -162.3808289, 162.5434418
15: -58.5512543, 50.2971840, -58.5126648, 50.3442764, -108.8955154, 108.8098450
16: -82.5441513, 65.9525757, -82.4839935, 65.8361588, -148.3802795, 148.4365692
17: -118.3656998, 77.6765900, -118.5476303, 77.5844498, -195.9501495, 196.2241974
18: -68.8158264, 41.8643951, -68.8629608, 41.5946426, -110.4104691, 110.7273331
19: -59.8413620, 24.8117428, -59.8324585, 24.5769024, -84.4182663, 84.6441956
20: -53.9359207, 32.1467857, -53.9211731, 32.0414352, -85.9773560, 86.0679550
21: -72.1483688, 36.3914795, -72.1086197, 36.1443672, -108.2927246, 108.5000916
22: -81.4637756, 47.6322403, -81.7127686, 47.5775223, -129.0412903, 129.3450012
23: -54.6300011, 34.5715332, -54.6797333, 34.3344498, -88.9644394, 89.2512665
24: -64.0729218, 34.5536346, -64.2365265, 34.5144272, -98.5873337, 98.7901611
25: -59.7967644, 39.3962860, -59.9034882, 39.3286934, -99.1254578, 99.2997742
26: -92.0872879, 49.9239578, -92.3397827, 49.8567085, -141.9439850, 142.2637329
27: -67.8224869, 44.1427956, -67.9511566, 44.0916824, -111.9141693, 112.0939407
28: -56.3365784, 36.4232330, -56.3906136, 36.3310242, -92.6676025, 92.8138428
29: -81.1640549, 53.7011604, -81.3230362, 53.5996552, -134.7637024, 135.0242004
30: -67.7355499, 36.6907234, -67.8137512, 36.5263977, -104.2619324, 104.5044708
31: -62.3444138, 30.5120697, -62.3682098, 30.3105469, -92.6549530, 92.8802795
32: -65.3183975, 47.6889191, -65.2851715, 47.7483597, -113.0667572, 112.9740906
33: -99.3187027, 58.2884216, -99.1933746, 58.1854630, -157.5041504, 157.4817963
34: -84.5944977, 44.3132057, -84.5590744, 44.2958374, -128.8903351, 128.8722839
35: -80.2236557, 47.2147713, -80.0520782, 47.1666489, -127.3902893, 127.2668381
36: -82.2053528, 48.3087120, -82.0334702, 48.2989922, -130.5043488, 130.3421783
37: -114.9272690, 47.9179688, -114.9781952, 47.8649750, -162.7922363, 162.8961639
38: -101.6583176, 63.3282051, -101.6031113, 63.3882065, -165.0465240, 164.9313202
39: -121.9424744, 54.5600052, -121.7916794, 54.6035156, -176.5459900, 176.3516846
40: -96.2483139, 47.2554169, -96.2866211, 47.3605003, -143.6088104, 143.5420227
41: -66.7624969, 39.6755219, -66.8204041, 39.7081871, -106.4706726, 106.4959259
42: -49.4917412, 44.3423119, -49.4774323, 44.1589699, -93.6507111, 93.8197479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
time: 101.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
time: 92.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -62.3998909, 45.8266525, -62.6543808, 46.1455154, -108.5454102, 108.4810257
1: -39.6724243, 41.7046547, -39.8391151, 41.8825531, -81.5549698, 81.5437622
2: -36.7745590, 43.6926041, -36.9756050, 43.9471817, -80.7217407, 80.6682129
3: -44.7273293, 51.7664909, -44.9245644, 52.1820488, -96.9093781, 96.6910553
4: -52.4212494, 40.4572372, -52.5685158, 40.6109314, -93.0321808, 93.0257568
5: -46.5304489, 56.7558098, -46.7677689, 57.1624908, -103.6929321, 103.5235748
6: -67.6169357, 41.3551712, -67.8083420, 41.6518021, -109.2687378, 109.1635132
7: -56.7737579, 52.6958694, -57.0574226, 52.9981003, -109.7718582, 109.7532883
8: -47.0091400, 46.9427719, -47.2349319, 47.1789093, -94.1880493, 94.1777039
9: -49.1032448, 52.2471161, -49.3408279, 52.5603371, -101.6635818, 101.5879440
10: -78.6380463, 76.0455551, -79.1924820, 76.4316483, -155.0697021, 155.2380371
11: -80.0045776, 52.6615753, -80.4082184, 52.8914604, -132.8960419, 133.0697937
12: -73.8632126, 58.0010910, -74.5535965, 58.6474495, -132.5106659, 132.5546875
13: -70.5386047, 66.0093689, -70.7946701, 66.6555328, -137.1941223, 136.8040466
14: -106.0621490, 56.5039215, -106.8222198, 57.0379066, -163.1000519, 163.3261414
15: -58.6410141, 50.3389702, -58.9445496, 50.5651588, -109.2061768, 109.2835236
16: -82.6090164, 66.0539169, -82.8612671, 66.2702560, -148.8792725, 148.9151917
17: -118.4152756, 77.9236603, -119.1587753, 78.5799255, -196.9952087, 197.0824280
18: -68.8701859, 41.9739876, -69.4010544, 42.0602608, -110.9304504, 111.3750305
19: -59.8778687, 24.8964272, -60.1975060, 24.9219933, -84.7998657, 85.0939331
20: -53.9716835, 32.2161636, -54.2731628, 32.3388786, -86.3105545, 86.4893265
21: -72.1880951, 36.5152206, -72.5795746, 36.6525650, -108.8406601, 109.0947952
22: -81.5054474, 47.7237854, -81.9814377, 47.9744225, -129.4798737, 129.7052307
23: -54.6616859, 34.6575394, -55.0198555, 34.6978798, -89.3595428, 89.6773911
24: -64.1123047, 34.5964279, -64.5290375, 34.7101364, -98.8224335, 99.1254578
25: -59.8303947, 39.4589844, -60.1546936, 39.6026115, -99.4330063, 99.6136780
26: -92.1284485, 50.0810127, -92.9181595, 50.5299835, -142.6584320, 142.9991760
27: -67.8841400, 44.1871490, -68.2774048, 44.3061523, -112.1902924, 112.4645462
28: -56.3700180, 36.4650116, -56.6640663, 36.5359650, -92.9059448, 93.1290741
29: -81.1974487, 53.8270950, -81.5704422, 54.1171303, -135.3145752, 135.3975220
30: -67.7717743, 36.7884178, -68.2051086, 36.9487228, -104.7204895, 104.9935303
31: -62.3988914, 30.5927467, -62.8279114, 30.6382637, -93.0371552, 93.4206543
32: -65.3654251, 47.7403679, -65.5300446, 47.9896088, -113.3550339, 113.2704010
33: -99.4652634, 58.3240356, -99.8157120, 58.6253242, -158.0905914, 158.1397400
34: -84.6884003, 44.3465385, -84.9788055, 44.5627213, -129.2511292, 129.3253479
35: -80.3570862, 47.2450523, -80.6139679, 47.5129204, -127.8700104, 127.8590240
36: -82.3044586, 48.3351822, -82.4545975, 48.5003510, -130.8048096, 130.7897797
37: -114.9988632, 47.9605370, -115.3549042, 48.0975418, -163.0964050, 163.3154449
38: -101.7516327, 63.3654747, -102.0419540, 63.6474304, -165.3990479, 165.4074249
39: -122.0680466, 54.5869751, -122.3571396, 54.9391861, -177.0072327, 176.9441223
40: -96.3360672, 47.2728806, -96.7126694, 47.5670853, -143.9031372, 143.9855347
41: -66.8092880, 39.7193260, -67.0639114, 39.9465256, -106.7558060, 106.7832336
42: -49.5257645, 44.4590492, -49.6752510, 44.6709442, -94.1966934, 94.1343002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1625
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
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
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2240894, upper bound: 57.1894843
time: 77.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3248741, upper bound: 57.1894843
time: 95.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.6703110, 46.0780449, -62.1889420, 45.8892822, -108.5595932, 108.2669830
1: -39.8661995, 41.8565979, -39.5854378, 41.7408676, -81.6070557, 81.4420242
2: -37.0201874, 43.9293175, -36.5761871, 43.7673569, -80.7875443, 80.5055008
3: -45.0030975, 52.0589447, -44.4695168, 51.8895874, -96.8926697, 96.5284576
4: -52.5728035, 40.5888214, -52.0875778, 40.4266891, -92.9994888, 92.6763992
5: -46.8465118, 57.0963974, -46.3531952, 56.9132347, -103.7597504, 103.4495850
6: -67.8106995, 41.5886993, -67.6052780, 41.4754562, -109.2861557, 109.1939774
7: -57.1434326, 52.9976082, -56.7295380, 52.9131737, -110.0565872, 109.7271423
8: -47.2510376, 47.1985245, -46.8816986, 47.0216331, -94.2726746, 94.0802155
9: -49.3920593, 52.5897560, -49.1319160, 52.3027267, -101.6947861, 101.7216721
10: -79.1503754, 76.5198517, -78.7576904, 75.5604401, -154.7108002, 155.2775269
11: -80.1400833, 52.8123398, -79.9294434, 52.1060867, -132.2461548, 132.7417908
12: -74.5460281, 58.6502304, -74.2207794, 57.8496933, -132.3957062, 132.8710022
13: -70.7923889, 66.3626862, -70.4379044, 66.2182159, -137.0105896, 136.8005676
14: -106.7788086, 57.0403938, -106.3668900, 56.3976517, -163.1764526, 163.4072876
15: -58.8675995, 50.5696411, -58.5576859, 50.3712387, -109.2388382, 109.1273270
16: -82.8118134, 66.3055344, -82.5301361, 65.8684845, -148.6802979, 148.8356628
17: -119.0122223, 78.4237747, -118.6945267, 77.5953064, -196.6075287, 197.1182861
18: -69.1467133, 42.1017113, -68.9127350, 41.6169205, -110.7636337, 111.0144501
19: -60.0535774, 24.9295559, -59.8669624, 24.5866623, -84.6402435, 84.7965164
20: -54.1626205, 32.3116570, -53.9584961, 32.0520248, -86.2146301, 86.2701492
21: -72.3995667, 36.6571579, -72.1521530, 36.1593399, -108.5589066, 108.8093033
22: -81.9073944, 48.0246925, -81.8059387, 47.5977478, -129.5051422, 129.8306274
23: -54.8699265, 34.6976852, -54.7117500, 34.3463554, -89.2162781, 89.4094315
24: -64.3512726, 34.6842270, -64.2602844, 34.5346031, -98.8858490, 98.9445114
25: -60.0087929, 39.6317596, -59.9371948, 39.3511162, -99.3599014, 99.5689468
26: -92.7830048, 50.6459732, -92.4988174, 49.8824883, -142.6654816, 143.1447906
27: -68.1808624, 44.2860641, -67.9752960, 44.1169357, -112.2977982, 112.2613602
28: -56.5691605, 36.5209503, -56.4180794, 36.3466873, -92.9158401, 92.9390259
29: -81.5277634, 54.1530228, -81.3966751, 53.6150932, -135.1428528, 135.5496826
30: -67.9978256, 36.9184189, -67.8413162, 36.5418777, -104.5396957, 104.7597275
31: -62.6342049, 30.6544743, -62.4009094, 30.3276310, -92.9618301, 93.0553741
32: -65.5188828, 47.9192696, -65.3130417, 47.7649994, -113.2838669, 113.2323074
33: -99.7402496, 58.4454384, -99.2204208, 58.2171211, -157.9573669, 157.6658630
34: -84.9741592, 44.5188828, -84.5841980, 44.3342934, -129.3084564, 129.1030884
35: -80.5988770, 47.3841476, -80.0736008, 47.1972580, -127.7961349, 127.4577408
36: -82.4821167, 48.4390564, -82.0623474, 48.3222580, -130.8043823, 130.5014038
37: -115.2698364, 48.1050797, -115.0138550, 47.8905907, -163.1604309, 163.1189270
38: -102.0812454, 63.5649834, -101.6312866, 63.4256592, -165.5068970, 165.1962585
39: -122.2641068, 54.7859383, -121.8186035, 54.6376266, -176.9017334, 176.6045380
40: -96.6538010, 47.5292397, -96.3101883, 47.4253349, -144.0791321, 143.8394165
41: -67.0440216, 39.9042130, -66.8412170, 39.7433853, -106.7874069, 106.7454300
42: -49.6603012, 44.6316071, -49.5030212, 44.1832161, -93.8435059, 94.1346283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1625
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1641
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
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
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
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2039522
time: 132.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2039522
time: 70.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.7758179, 46.1060944, -62.6843338, 46.2064629, -108.9822769, 108.7904282
1: -39.9258575, 41.8805771, -39.8564644, 41.9078674, -81.8337250, 81.7370377
2: -37.1200027, 43.9486618, -36.9923096, 44.0075378, -81.1275330, 80.9409714
3: -45.1180840, 52.0859222, -44.9441452, 52.2513428, -97.3694305, 97.0300598
4: -52.6920586, 40.6129913, -52.5856323, 40.6363640, -93.3284225, 93.1986160
5: -46.9503555, 57.1216240, -46.7905045, 57.2475433, -104.1978989, 103.9121246
6: -67.8531952, 41.6259003, -67.8335724, 41.6898689, -109.5430603, 109.4594727
7: -57.2215576, 53.0186081, -57.0870628, 53.0637207, -110.2852783, 110.1056595
8: -47.3370667, 47.2260857, -47.2481499, 47.2357712, -94.5728378, 94.4742355
9: -49.4465675, 52.6476364, -49.4102707, 52.5783081, -102.0248718, 102.0578995
10: -79.2024002, 76.7384186, -79.3157501, 76.4572601, -155.6596680, 156.0541382
11: -80.1825867, 53.0199242, -80.4514771, 52.9053345, -133.0879211, 133.4714050
12: -74.5764618, 58.8509750, -74.7280426, 58.6677780, -133.2442322, 133.5790100
13: -70.8826141, 66.4203339, -70.8606415, 66.6831970, -137.5657959, 137.2809753
14: -106.8447952, 57.2006912, -106.9895630, 57.0503654, -163.8951416, 164.1902466
15: -58.9559021, 50.6117020, -58.9888229, 50.5922852, -109.5481873, 109.6005096
16: -82.8769379, 66.4083252, -82.9075775, 66.3020477, -149.1789856, 149.3159027
17: -119.0615311, 78.6710510, -119.3056564, 78.5908203, -197.6523285, 197.9767151
18: -69.2012558, 42.2113800, -69.4514160, 42.0824127, -111.2836685, 111.6627960
19: -60.0901184, 25.0142517, -60.2321587, 24.9318066, -85.0219269, 85.2464142
20: -54.1983337, 32.3811378, -54.3109589, 32.3494034, -86.5477219, 86.6920929
21: -72.4391479, 36.7809868, -72.6233215, 36.6674995, -109.1066437, 109.4043121
22: -81.9489899, 48.1166916, -82.0749588, 47.9945030, -129.9434814, 130.1916504
23: -54.9019547, 34.7837639, -55.0522232, 34.7096634, -89.6116180, 89.8359833
24: -64.3910217, 34.7272644, -64.5535278, 34.7302399, -99.1212616, 99.2807846
25: -60.0425644, 39.6946182, -60.1885300, 39.6249161, -99.6674728, 99.8831406
26: -92.8237152, 50.8038330, -93.0771103, 50.5556679, -143.3793793, 143.8809357
27: -68.2427597, 44.3305893, -68.3020401, 44.3309479, -112.5737076, 112.6326294
28: -56.6028442, 36.5627632, -56.6917686, 36.5515366, -93.1543808, 93.2545319
29: -81.5607758, 54.2793007, -81.6440887, 54.1324158, -135.6931763, 135.9233704
30: -68.0343628, 37.0159950, -68.2328491, 36.9641037, -104.9984360, 105.2488403
31: -62.6899338, 30.7351665, -62.8612099, 30.6553116, -93.3452454, 93.5963745
32: -65.5660934, 47.9709969, -65.5578461, 48.0062866, -113.5723801, 113.5288391
33: -99.8868942, 58.4808655, -99.8426666, 58.6570663, -158.5439453, 158.3235321
34: -85.0683441, 44.5519485, -85.0040131, 44.6013832, -129.6697235, 129.5559692
35: -80.7324066, 47.4140739, -80.6354599, 47.5439034, -128.2763062, 128.0495300
36: -82.5817871, 48.4660645, -82.4831924, 48.5240593, -131.1058350, 130.9492493
37: -115.3417969, 48.1477165, -115.3906097, 48.1232033, -163.4649963, 163.5383148
38: -102.1753693, 63.6023712, -102.0700607, 63.6850052, -165.8603821, 165.6724243
39: -122.3897324, 54.8130074, -122.3836060, 54.9735184, -177.3632507, 177.1965942
40: -96.7416992, 47.5466805, -96.7360764, 47.6321449, -144.3738251, 144.2827606
41: -67.0912476, 39.9471817, -67.0847321, 39.9801178, -107.0713654, 107.0319138
42: -49.6948509, 44.7489853, -49.7011032, 44.6949005, -94.3897552, 94.4500809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 908
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2240894, upper bound: 57.2605813
time: 80.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.3248741, upper bound: 57.2605813
time: 77.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.4677200, 45.8371239, -62.7345963, 46.0492325, -108.5169449, 108.5717163
1: -39.7176132, 41.7131500, -39.9076729, 41.8527184, -81.5703201, 81.6208191
2: -36.8552895, 43.6975555, -37.1461258, 43.8827133, -80.7380066, 80.8436737
3: -44.7992783, 51.7782784, -45.0559998, 52.0491943, -96.8484726, 96.8342743
4: -52.5038757, 40.4691315, -52.7232132, 40.5859642, -93.0898285, 93.1923447
5: -46.6145706, 56.7653275, -46.9360275, 57.0752068, -103.6897736, 103.7013474
6: -67.6376419, 41.4193802, -67.8104248, 41.7715111, -109.4091339, 109.2298050
7: -56.8470955, 52.7086296, -57.2110214, 53.0067024, -109.8537903, 109.9196472
8: -47.1023788, 46.9497910, -47.4502411, 47.1543655, -94.2567215, 94.4000244
9: -49.1087608, 52.3409195, -49.2989540, 52.7827377, -101.8914871, 101.6398621
10: -78.6646576, 76.1754456, -79.0782928, 76.6520538, -155.3166809, 155.2537231
11: -80.0204010, 52.7105789, -80.1441193, 52.9068375, -132.9272156, 132.8547058
12: -73.8760376, 58.1721115, -74.4185181, 59.0412292, -132.9172668, 132.5906372
13: -70.5080719, 66.0461502, -70.5701141, 66.5386887, -137.0467529, 136.6162720
14: -106.0906372, 56.5728607, -106.6693802, 57.1416550, -163.2322998, 163.2422333
15: -58.7066193, 50.3588448, -59.0197449, 50.5744934, -109.2811127, 109.3785858
16: -82.6286163, 66.1378784, -82.8019867, 66.4290009, -149.0576172, 148.9398651
17: -118.4287186, 78.0000763, -118.9062576, 78.6371689, -197.0658875, 196.9063416
18: -68.8899841, 41.9840698, -69.1816711, 41.9862709, -110.8762512, 111.1657410
19: -59.8921127, 24.8956375, -60.0585594, 24.8381863, -84.7303009, 84.9541931
20: -53.9888420, 32.2307930, -54.1554756, 32.3152542, -86.3040848, 86.3862610
21: -72.2027893, 36.5292740, -72.3914795, 36.5944748, -108.7972565, 108.9207458
22: -81.5350342, 47.7523613, -81.9796219, 47.9736557, -129.5086823, 129.7319794
23: -54.6765823, 34.6555328, -54.8754158, 34.6075211, -89.2840881, 89.5309448
24: -64.1585388, 34.5857658, -64.5314484, 34.6156197, -98.7741547, 99.1172180
25: -59.8429565, 39.4694519, -60.0776825, 39.5711403, -99.4140778, 99.5471344
26: -92.1497955, 50.1706772, -92.7217331, 50.6726418, -142.8224335, 142.8924103
27: -67.9420013, 44.1735229, -68.3572311, 44.1983948, -112.1403885, 112.5307388
28: -56.3841209, 36.4640503, -56.5770874, 36.4695473, -92.8536682, 93.0411377
29: -81.2205276, 53.8602829, -81.5351105, 54.1073647, -135.3278809, 135.3953857
30: -67.7845001, 36.8177757, -68.0102081, 36.9426880, -104.7271881, 104.8279877
31: -62.4269791, 30.5751152, -62.7031746, 30.5104485, -92.9374237, 93.2782898
32: -65.3800507, 47.8036079, -65.4996338, 48.1210327, -113.5010834, 113.3032379
33: -99.4975510, 58.3448563, -99.7853394, 58.4592171, -157.9567566, 158.1301880
34: -84.7111816, 44.3630753, -84.9497070, 44.5109406, -129.2221222, 129.3127747
35: -80.3885345, 47.2602654, -80.5851517, 47.3840294, -127.7725525, 127.8454132
36: -82.2987976, 48.3502808, -82.3470917, 48.4632568, -130.7620544, 130.6973572
37: -115.0256119, 47.9826736, -115.3157806, 48.0968437, -163.1224518, 163.2984619
38: -101.7934570, 63.3840637, -102.0502777, 63.5956879, -165.3891144, 165.4343414
39: -122.0762787, 54.6013336, -122.2415161, 54.7767868, -176.8530579, 176.8428497
40: -96.3775711, 47.2827110, -96.7285461, 47.4979210, -143.8754730, 144.0112610
41: -66.8322449, 39.7591438, -67.0671539, 39.9809761, -106.8132172, 106.8262939
42: -49.5424461, 44.5424767, -49.6618118, 44.8037033, -94.3461456, 94.2042847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1479
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
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 680
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
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
time: 117.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2566793
time: 92.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 212.59 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2240894, upper bound: 57.1894843
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.3248741, upper bound: 57.1894843
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2039522
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2039522
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2240894, upper bound: 57.2605813
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.3248741, upper bound: 57.2605813
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.1251692
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 212.59
Output dim: 2, lower bound: -57.2160129, upper bound: 57.2566793
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2114688
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3489547
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.1464764
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2114688
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.4244357, upper bound: 57.2797078
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2685495
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.3585403
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.2212445
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 212.59
Output dim: 2, lower bound: -57.3269217, upper bound: 57.4376493
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=81.74913024902344
rel_dist={2: [-57.47217120413072, 57.472171209697024]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8722250, upper bound: 53.9517478
time: 71.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8722250, upper bound: 53.9517478
time: 85.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 156.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 156.92
Output dim: 2, lower bound: -53.8722250, upper bound: 53.9517478
IS_A2, status: Status.UNKNOWN, split count: 1, time: 156.92
Output dim: 2, lower bound: -53.8722250, upper bound: 53.9517478

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -63.0016670, 46.1784630, -63.1703682, 46.2247086, -109.2263718, 109.3488312
1: -40.0608215, 41.9438629, -40.1584358, 41.9809952, -82.0418015, 82.1022949
2: -37.3429146, 43.9967194, -37.5271530, 44.0240822, -81.3669968, 81.5238724
3: -45.3487587, 52.1926460, -45.5175781, 52.2410355, -97.5897980, 97.7102203
4: -52.9470558, 40.6721992, -53.1720505, 40.7136612, -93.6607208, 93.8442459
5: -47.1742783, 57.2239685, -47.3485146, 57.2716751, -104.4459534, 104.5724792
6: -67.9469528, 41.8579330, -68.0131683, 42.0257568, -109.9727097, 109.8711014
7: -57.4120445, 53.1295624, -57.5456848, 53.1771278, -110.5891724, 110.6752472
8: -47.5625572, 47.2853432, -47.7660141, 47.3267250, -94.8892822, 95.0513535
9: -49.5510902, 52.8360023, -49.6144562, 53.0076218, -102.5587158, 102.4504547
10: -79.3242340, 77.1661530, -79.4083099, 77.5040817, -156.8283081, 156.5744629
11: -80.2767029, 53.3444786, -80.3540192, 53.6555977, -133.9322968, 133.6985016
12: -74.6675873, 59.2950630, -74.7248764, 59.7437820, -134.4113770, 134.0199280
13: -71.0004272, 66.5607605, -71.0581665, 66.7505798, -137.7510071, 137.6189270
14: -107.0077667, 57.4680252, -107.1106110, 57.7367630, -164.7445374, 164.5786438
15: -59.2739868, 50.7019501, -59.4750175, 50.7719955, -110.0459824, 110.1769714
16: -83.0029984, 66.6599503, -83.1080017, 66.8642197, -149.8672180, 149.7679443
17: -119.1782837, 79.0953064, -119.2546082, 79.5342865, -198.7125397, 198.3498840
18: -69.3224792, 42.3745041, -69.4397736, 42.4635735, -111.7860565, 111.8142776
19: -60.1685791, 25.1261559, -60.2336807, 25.1941433, -85.3627167, 85.3598328
20: -54.2787437, 32.4904060, -54.3376007, 32.5884819, -86.8672180, 86.8280029
21: -72.5295181, 36.9514503, -72.5969543, 37.0898819, -109.6194000, 109.5484009
22: -82.1464233, 48.2740059, -82.2576828, 48.3761063, -130.5225067, 130.5316772
23: -54.9721603, 34.8977737, -55.0318527, 34.9690590, -89.9412231, 89.9296265
24: -64.5143280, 34.7826767, -64.6805115, 34.8180962, -99.3324051, 99.4631805
25: -60.1498032, 39.7958908, -60.2395325, 39.8583832, -100.0081863, 100.0354156
26: -92.9877014, 51.0992432, -93.0670700, 51.3108292, -144.2985229, 144.1663055
27: -68.4123840, 44.3966713, -68.5979080, 44.4305649, -112.8429337, 112.9945831
28: -56.6728668, 36.6275101, -56.7330055, 36.6675682, -93.3404236, 93.3605118
29: -81.6733551, 54.4776573, -81.7530365, 54.6251335, -136.2984924, 136.2306824
30: -68.1156845, 37.1841164, -68.1835480, 37.3227386, -105.4384232, 105.3676605
31: -62.8120842, 30.8302631, -62.9396935, 30.8730869, -93.6851654, 93.7699509
32: -65.6613159, 48.1253128, -65.7275085, 48.3017578, -113.9630737, 113.8528214
33: -100.1179428, 58.5664062, -100.3463821, 58.6298103, -158.7477570, 158.9127808
34: -85.2263031, 44.6338501, -85.3494492, 44.6868324, -129.9131317, 129.9832916
35: -80.9530182, 47.4830894, -81.1266479, 47.5322418, -128.4852600, 128.6097412
36: -82.7201691, 48.5291252, -82.7968292, 48.5886116, -131.3087769, 131.3259583
37: -115.5018463, 48.2389297, -115.6455765, 48.2974243, -163.7992554, 163.8845062
38: -102.3634338, 63.6911469, -102.4799347, 63.7764969, -166.1399231, 166.1710815
39: -122.6005096, 54.8789597, -122.7749481, 54.9218292, -177.5223389, 177.6539001
40: -96.9164505, 47.6048622, -97.0867462, 47.6400070, -144.5564575, 144.6916046
41: -67.1906281, 40.1013680, -67.2754517, 40.2174721, -107.4080963, 107.3768158
42: -49.7730141, 45.0194016, -49.8294868, 45.2582474, -95.0312653, 94.8488922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=372, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
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
type: B, layer: 1, pos: 1400
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
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 725
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
Output dim: 2, lower bound: -53.8522947, upper bound: 53.8362906
time: 74.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8522947, upper bound: 53.9487047
time: 72.20 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -63.3884659, 46.4092941, -63.3082848, 46.2619400, -109.6504059, 109.7175751
1: -40.2785645, 42.0561752, -40.2328644, 42.0093079, -82.2878723, 82.2890320
2: -37.7138939, 44.2348862, -37.6781464, 44.0456696, -81.7595520, 81.9130325
3: -45.6939659, 52.4097023, -45.6552505, 52.2799530, -97.9739227, 98.0649567
4: -53.4087982, 40.9603653, -53.3566399, 40.7459679, -94.1547699, 94.3169937
5: -47.5300598, 57.5281029, -47.4928360, 57.3101501, -104.8401947, 105.0209351
6: -68.2390747, 42.1814842, -68.0655518, 42.1451073, -110.3841858, 110.2470245
7: -57.7241211, 53.2586327, -57.6497765, 53.2103615, -110.9344559, 110.9084091
8: -47.9823532, 47.5534782, -47.9329491, 47.3589554, -95.3412933, 95.4864273
9: -49.7788239, 53.2098503, -49.6639404, 53.1478729, -102.9266968, 102.8737946
10: -79.8044739, 77.8485413, -79.4748077, 77.7825775, -157.5870209, 157.3233490
11: -80.6970978, 53.9490051, -80.4163513, 53.9122238, -134.6093140, 134.3653564
12: -75.2563324, 60.1641998, -74.7703400, 60.1163406, -135.3726807, 134.9345398
13: -71.2097778, 67.0046844, -71.1028748, 66.9043121, -138.1140900, 138.1075439
14: -107.4735489, 57.9991798, -107.1905670, 57.9664993, -165.4400330, 165.1897430
15: -59.6784286, 51.0344505, -59.6139526, 50.8288422, -110.5072708, 110.6484070
16: -83.3799133, 67.0744476, -83.1917801, 67.0232239, -150.4031372, 150.2662354
17: -119.6977081, 79.9486084, -119.3136978, 79.9027939, -199.6004944, 199.2622986
18: -69.6526642, 42.5886497, -69.5143890, 42.5326538, -112.1853180, 112.1030273
19: -60.4160500, 25.2779121, -60.2835960, 25.2488766, -85.6649246, 85.5615082
20: -54.5291977, 32.6928902, -54.3853722, 32.6680794, -87.1972809, 87.0782471
21: -72.8720551, 37.2309341, -72.6507721, 37.2028122, -110.0748672, 109.8817062
22: -82.3923416, 48.5531044, -82.3182144, 48.4588623, -130.8511963, 130.8713226
23: -55.1852112, 35.0653648, -55.0800095, 35.0264740, -90.2116852, 90.1453705
24: -64.8885498, 34.9694977, -64.8166885, 34.8446884, -99.7332306, 99.7861786
25: -60.3670120, 39.9802551, -60.3008385, 39.9082184, -100.2752304, 100.2810822
26: -93.2987366, 51.5121956, -93.1300125, 51.4697304, -144.7684631, 144.6422119
27: -68.8355484, 44.5341835, -68.7496796, 44.4557266, -113.2912674, 113.2838593
28: -56.8597832, 36.7313995, -56.7796211, 36.6972427, -93.5570221, 93.5110168
29: -81.8847046, 54.7821350, -81.7998810, 54.7419052, -136.6266174, 136.5820007
30: -68.3390198, 37.4886322, -68.2368698, 37.4363861, -105.7753906, 105.7255020
31: -63.1854401, 30.9387474, -63.0448875, 30.9049435, -94.0903854, 93.9836273
32: -65.9957962, 48.4745941, -65.7801361, 48.4464760, -114.4422760, 114.2547302
33: -100.6208496, 58.8414917, -100.5344391, 58.6789398, -159.2997894, 159.3759308
34: -85.5063934, 44.8319778, -85.4496002, 44.7262039, -130.2326050, 130.2815857
35: -81.3342209, 47.6985779, -81.2688141, 47.5713921, -128.9056091, 128.9673920
36: -82.9317322, 48.6838875, -82.8513184, 48.6330872, -131.5648041, 131.5352020
37: -115.8673401, 48.4071426, -115.7555542, 48.3436775, -164.2110138, 164.1626892
38: -102.6810760, 63.9082031, -102.5681534, 63.8435516, -166.5246277, 166.4763489
39: -123.0573502, 55.0508804, -122.9220505, 54.9535675, -178.0109100, 177.9729309
40: -97.3252945, 47.7326355, -97.2239609, 47.6606979, -144.9859924, 144.9565887
41: -67.4661102, 40.3551216, -67.3438263, 40.3123245, -107.7784348, 107.6989365
42: -50.0523376, 45.4968834, -49.8745155, 45.4551163, -95.5074463, 95.3713989

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1670
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1560
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8522947, upper bound: 53.8362906
time: 78.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8522947, upper bound: 53.9487047
time: 72.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 153.78 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 153.78
Output dim: 2, lower bound: -53.8522947, upper bound: 53.8362906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 153.78
Output dim: 2, lower bound: -53.8522947, upper bound: 53.9487047
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 153.78
Output dim: 2, lower bound: -53.8522947, upper bound: 53.8362906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 153.78
Output dim: 2, lower bound: -53.8522947, upper bound: 53.9487047

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -62.7072906, 46.1134605, -62.5578232, 45.9965439, -108.7038345, 108.6712799
1: -39.8824921, 41.8889465, -39.7924271, 41.8375626, -81.7200546, 81.6813736
2: -37.0368576, 43.9554596, -36.9050865, 43.8437881, -80.8806458, 80.8605423
3: -45.0310783, 52.1266708, -44.8727570, 52.0044708, -97.0355377, 96.9994125
4: -52.6048203, 40.6107025, -52.4751854, 40.5198212, -93.1246414, 93.0858917
5: -46.8608589, 57.1646461, -46.7066269, 57.0184059, -103.8792648, 103.8712692
6: -67.8394318, 41.6774445, -67.7678375, 41.6412239, -109.4806442, 109.4452744
7: -57.1532822, 53.0718536, -57.0008087, 53.0113678, -110.1646500, 110.0726624
8: -47.2587891, 47.2265816, -47.1458435, 47.1294823, -94.3882751, 94.3724213
9: -49.4491692, 52.5784760, -49.3622704, 52.4748421, -101.9240112, 101.9407501
10: -79.1909027, 76.5751648, -78.9477997, 76.3134003, -155.5042877, 155.5229645
11: -80.1767426, 52.9094505, -80.0826950, 52.7919388, -132.9686890, 132.9921417
12: -74.5941162, 58.6657906, -74.3419189, 58.4684830, -133.0625916, 133.0076904
13: -70.8956223, 66.4000397, -70.8267212, 66.3844833, -137.2801056, 137.2267456
14: -106.8477249, 57.0799713, -106.6224442, 56.9423218, -163.7900391, 163.7024231
15: -59.0072098, 50.5963097, -58.9091034, 50.5279770, -109.5351868, 109.5054092
16: -82.8580780, 66.3410568, -82.7697296, 66.2066956, -149.0647583, 149.1107788
17: -119.0718689, 78.5474930, -118.8822174, 78.4247284, -197.4965973, 197.4297180
18: -69.1966705, 42.1720085, -69.1042786, 42.0438004, -111.2404709, 111.2762909
19: -60.0825043, 24.9860516, -59.9970741, 24.9121780, -84.9946747, 84.9831238
20: -54.1890640, 32.3474426, -54.0935249, 32.2977676, -86.4868317, 86.4409637
21: -72.4371948, 36.7175598, -72.3011780, 36.6133385, -109.0505219, 109.0187302
22: -82.0258636, 48.0678406, -81.9733276, 47.9432335, -129.9691010, 130.0411682
23: -54.8921776, 34.7549973, -54.8263550, 34.6754990, -89.5676727, 89.5813446
24: -64.3670502, 34.7271385, -64.3618927, 34.7097702, -99.0768204, 99.0890198
25: -60.0708427, 39.6711426, -60.0531235, 39.5947380, -99.6655731, 99.7242661
26: -92.8826065, 50.6781540, -92.6720886, 50.4497375, -143.3323364, 143.3502502
27: -68.2084885, 44.3439293, -68.1613998, 44.3129349, -112.5214081, 112.5053253
28: -56.5915413, 36.5571938, -56.5365906, 36.5115814, -93.1031189, 93.0937805
29: -81.5776138, 54.2067566, -81.5252838, 54.0817795, -135.6593933, 135.7320404
30: -68.0310516, 36.9687958, -67.9753418, 36.8797226, -104.9107513, 104.9441376
31: -62.6672859, 30.7223091, -62.5835228, 30.6525364, -93.3198242, 93.3058167
32: -65.5559006, 47.9307632, -65.4973450, 47.9066315, -113.4625320, 113.4281006
33: -99.8142319, 58.4710464, -99.7228012, 58.3457336, -158.1599731, 158.1938477
34: -85.0273438, 44.5496902, -84.9359131, 44.4604836, -129.4878235, 129.4855957
35: -80.6725769, 47.4068222, -80.5617828, 47.3058968, -127.9784698, 127.9686050
36: -82.5579529, 48.4568939, -82.4546280, 48.4144745, -130.9724274, 130.9115143
37: -115.3333893, 48.1286736, -115.2778473, 48.0481911, -163.3815765, 163.4065247
38: -102.1313705, 63.5950394, -101.9937210, 63.5505028, -165.6818695, 165.5887451
39: -122.3733063, 54.8083344, -122.2911682, 54.7395477, -177.1128540, 177.0994873
40: -96.6966553, 47.5581055, -96.6161423, 47.4957695, -144.1924286, 144.1742554
41: -67.0705261, 39.9598427, -67.0104218, 39.9081268, -106.9786530, 106.9702606
42: -49.6856537, 44.6792336, -49.6328430, 44.5701447, -94.2557983, 94.3120728

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 618
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
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1393
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
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1368
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
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
time: 77.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189
time: 77.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -62.9911652, 46.1759300, -63.1454735, 46.2189026, -109.2100677, 109.3214035
1: -40.0537033, 41.9417877, -40.1419830, 41.9760284, -82.0297318, 82.0837631
2: -37.3332710, 43.9949265, -37.5042305, 44.0198975, -81.3531647, 81.4991608
3: -45.3377953, 52.1897507, -45.4913902, 52.2341423, -97.5719376, 97.6811371
4: -52.9359703, 40.6692047, -53.1461182, 40.7066307, -93.6426010, 93.8153152
5: -47.1636276, 57.2212486, -47.3228378, 57.2653427, -104.4289627, 104.5440826
6: -67.9420624, 41.8428383, -68.0016937, 41.9888229, -109.9308777, 109.8445282
7: -57.4024391, 53.1273003, -57.5228729, 53.1717453, -110.5741882, 110.6501770
8: -47.5520859, 47.2828636, -47.7414360, 47.3208504, -94.8729324, 95.0242996
9: -49.5470924, 52.8271561, -49.6051712, 52.9865837, -102.5336761, 102.4323273
10: -79.3188248, 77.1462631, -79.3955917, 77.4577484, -156.7765808, 156.5418549
11: -80.2722321, 53.3314285, -80.3439178, 53.6246834, -133.8969116, 133.6753540
12: -74.6639404, 59.2765274, -74.7162323, 59.6993828, -134.3633270, 133.9927521
13: -70.9893799, 66.5546341, -71.0321808, 66.7362823, -137.7256622, 137.5868225
14: -107.0012360, 57.4560165, -107.0950699, 57.7076530, -164.7088928, 164.5510864
15: -59.2572327, 50.6972466, -59.4354706, 50.7609024, -110.0181198, 110.1327133
16: -82.9962463, 66.6470184, -83.0922470, 66.8357849, -149.8320312, 149.7392578
17: -119.1738129, 79.0792389, -119.2438889, 79.4961624, -198.6699829, 198.3231201
18: -69.3174744, 42.3668823, -69.4279633, 42.4454956, -111.7629700, 111.7948456
19: -60.1653595, 25.1208382, -60.2260742, 25.1817188, -85.3470764, 85.3469086
20: -54.2752724, 32.4856415, -54.3295174, 32.5772324, -86.8525085, 86.8151550
21: -72.5253906, 36.9439697, -72.5873566, 37.0719528, -109.5973434, 109.5313263
22: -82.1413727, 48.2626686, -82.2455902, 48.3490982, -130.4904785, 130.5082550
23: -54.9690323, 34.8925705, -55.0243759, 34.9571877, -89.9262238, 89.9169388
24: -64.5070190, 34.7798462, -64.6637268, 34.8114929, -99.3185120, 99.4435577
25: -60.1458321, 39.7898598, -60.2298393, 39.8443985, -99.9902191, 100.0196915
26: -92.9833755, 51.0852699, -93.0569382, 51.2768402, -144.2601929, 144.1422119
27: -68.4036026, 44.3936081, -68.5775146, 44.4233017, -112.8269043, 112.9711227
28: -56.6698837, 36.6225204, -56.7260590, 36.6556091, -93.3254776, 93.3485718
29: -81.6685944, 54.4675751, -81.7420120, 54.6011505, -136.2697449, 136.2095642
30: -68.1118698, 37.1770020, -68.1747894, 37.3057823, -105.4176483, 105.3517914
31: -62.8060036, 30.8243198, -62.9247246, 30.8592300, -93.6652374, 93.7490463
32: -65.6564255, 48.1203079, -65.7161865, 48.2898102, -113.9462357, 113.8364868
33: -100.1082001, 58.5626793, -100.3228912, 58.6213837, -158.7295837, 158.8855591
34: -85.2194366, 44.6301422, -85.3328781, 44.6781235, -129.8975372, 129.9630127
35: -80.9438019, 47.4800491, -81.1049423, 47.5252533, -128.4690552, 128.5849915
36: -82.7134857, 48.5261002, -82.7812500, 48.5814743, -131.2949524, 131.3073425
37: -115.4934845, 48.2332306, -115.6257248, 48.2838058, -163.7772827, 163.8589478
38: -102.3542633, 63.6853027, -102.4584579, 63.7629013, -166.1171570, 166.1437683
39: -122.5904083, 54.8760109, -122.7508698, 54.9149857, -177.5054016, 177.6268768
40: -96.9079132, 47.6022568, -97.0667191, 47.6338120, -144.5417175, 144.6689758
41: -67.1854858, 40.0897675, -67.2633667, 40.1896896, -107.3751678, 107.3531265
42: -49.7689629, 45.0089340, -49.8199387, 45.2337532, -95.0027008, 94.8288727

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
type: A, layer: 1, pos: 1671
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
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1631
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
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
time: 316.48 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.9444702
time: 200.18 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.0940094, 46.3450432, -62.6953926, 46.0343628, -109.1283569, 109.0404358
1: -40.1005630, 42.0010986, -39.8670120, 41.8661804, -81.9667358, 81.8681107
2: -37.4080505, 44.1939926, -37.0561218, 43.8656464, -81.2736969, 81.2501068
3: -45.3761826, 52.3442726, -45.0102806, 52.0440445, -97.4202271, 97.3545532
4: -53.0668526, 40.8992119, -52.6598206, 40.5523262, -93.6191788, 93.5590210
5: -47.2166672, 57.4692993, -46.8508453, 57.0573692, -104.2740326, 104.3201370
6: -68.1323395, 41.9977341, -67.8206329, 41.7594337, -109.8917694, 109.8183594
7: -57.4645920, 53.2009544, -57.1049805, 53.0450783, -110.5096588, 110.3059311
8: -47.6786537, 47.4951782, -47.3126640, 47.1620140, -94.8406677, 94.8078384
9: -49.6782188, 52.9522247, -49.4127426, 52.6147652, -102.2929840, 102.3649673
10: -79.6725311, 77.2578964, -79.0151672, 76.5920944, -156.2646179, 156.2730408
11: -80.5990295, 53.5140648, -80.1457672, 53.0484543, -133.6474915, 133.6598206
12: -75.1835175, 59.5349464, -74.3878479, 58.8408699, -134.0243835, 133.9227905
13: -71.1057587, 66.8426437, -70.8718109, 66.5382385, -137.6439819, 137.7144470
14: -107.3144226, 57.6104164, -106.7032089, 57.1714363, -164.4858398, 164.3136292
15: -59.4016304, 50.9295158, -59.0383224, 50.5853806, -109.9870148, 109.9678345
16: -83.2361298, 66.7501373, -82.8552628, 66.3606491, -149.5967560, 149.6054077
17: -119.5918579, 79.3999786, -118.9418106, 78.7925568, -198.3844147, 198.3417816
18: -69.5258026, 42.3848648, -69.1791153, 42.1112900, -111.6370850, 111.5639801
19: -60.3296661, 25.1381607, -60.0476875, 24.9670067, -85.2966766, 85.1858368
20: -54.4402504, 32.5498810, -54.1419983, 32.3772430, -86.8174896, 86.6918793
21: -72.7805481, 36.9972992, -72.3556519, 36.7264290, -109.5069656, 109.3529510
22: -82.2713623, 48.3482857, -82.0340195, 48.0262337, -130.2975922, 130.3823090
23: -55.1055527, 34.9228477, -54.8748741, 34.7330437, -89.8385773, 89.7977142
24: -64.7404251, 34.9147415, -64.4966583, 34.7365570, -99.4769821, 99.4113998
25: -60.2878914, 39.8564796, -60.1144447, 39.6445618, -99.9324417, 99.9709244
26: -93.1934586, 51.0910568, -92.7355652, 50.6087189, -143.8021851, 143.8266296
27: -68.6307907, 44.4819870, -68.3120804, 44.3381882, -112.9689789, 112.7940598
28: -56.7782516, 36.6611557, -56.5831070, 36.5413589, -93.3196030, 93.2442551
29: -81.7885895, 54.5117874, -81.5717926, 54.1984520, -135.9870453, 136.0835876
30: -68.2541656, 37.2730789, -68.0290222, 36.9932251, -105.2473907, 105.3020935
31: -63.0382767, 30.8310280, -62.6869049, 30.6844749, -93.7227478, 93.5179291
32: -65.8922043, 48.2798767, -65.5503387, 48.0512009, -113.9434052, 113.8302078
33: -100.3167648, 58.7467995, -99.9104462, 58.3958359, -158.7126007, 158.6572418
34: -85.3068237, 44.7482071, -85.0355225, 44.5004196, -129.8072357, 129.7837219
35: -81.0534668, 47.6232491, -80.7036057, 47.3457870, -128.3992615, 128.3268433
36: -82.7694092, 48.6114502, -82.5090790, 48.4595718, -131.2289734, 131.1205292
37: -115.6982803, 48.2971420, -115.3874893, 48.0946655, -163.7929382, 163.6846313
38: -102.4484406, 63.8111992, -102.0814972, 63.6164017, -166.0648499, 165.8927002
39: -122.8286896, 54.9803009, -122.4369354, 54.7715874, -177.6002808, 177.4172363
40: -97.1049957, 47.6856422, -96.7528305, 47.5170517, -144.6220398, 144.4384613
41: -67.3462906, 40.2096786, -67.0790405, 40.0013275, -107.3476181, 107.2887115
42: -49.9661140, 45.1562653, -49.6782188, 44.7666702, -94.7327881, 94.8344879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
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
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 930
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
type: A, layer: 1, pos: 1368
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
time: 103.03 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189
time: 81.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.3779488, 46.4067383, -63.2833214, 46.2561188, -109.6340637, 109.6900635
1: -40.2714767, 42.0540924, -40.2163696, 42.0042992, -82.2757721, 82.2704620
2: -37.7042923, 44.2330780, -37.6552010, 44.0414581, -81.7457428, 81.8882751
3: -45.6830177, 52.4067993, -45.6289597, 52.2730484, -97.9560623, 98.0357513
4: -53.3977165, 40.9573746, -53.3306122, 40.7389336, -94.1366501, 94.2879791
5: -47.5194054, 57.5253792, -47.4670792, 57.3037872, -104.8231964, 104.9924545
6: -68.2342072, 42.1660576, -68.0540314, 42.1077423, -110.3419342, 110.2200928
7: -57.7144585, 53.2563667, -57.6268806, 53.2049446, -110.9194031, 110.8832321
8: -47.9718666, 47.5509987, -47.9082680, 47.3530426, -95.3248978, 95.4592590
9: -49.7748566, 53.2010384, -49.6546783, 53.1268387, -102.9016953, 102.8557129
10: -79.7991028, 77.8286972, -79.4620285, 77.7362366, -157.5352936, 157.2907257
11: -80.6927338, 53.9359703, -80.4062653, 53.8812180, -134.5739441, 134.3422241
12: -75.2526779, 60.1457405, -74.7616196, 60.0718155, -135.3244934, 134.9073639
13: -71.1986237, 66.9985046, -71.0768204, 66.8899078, -138.0885315, 138.0753174
14: -107.4670563, 57.9871330, -107.1749954, 57.9372406, -165.4042969, 165.1621246
15: -59.6627350, 51.0297508, -59.5764580, 50.8177414, -110.4804764, 110.6062088
16: -83.3731613, 67.0615387, -83.1760254, 66.9958572, -150.3690186, 150.2375641
17: -119.6932220, 79.9325714, -119.3029099, 79.8646240, -199.5578308, 199.2354736
18: -69.6475983, 42.5809975, -69.5025330, 42.5144501, -112.1620483, 112.0835266
19: -60.4128227, 25.2726231, -60.2759933, 25.2364464, -85.6492538, 85.5486145
20: -54.5257607, 32.6881027, -54.3773003, 32.6567841, -87.1825409, 87.0653992
21: -72.8679962, 37.2234726, -72.6411743, 37.1848412, -110.0528412, 109.8646469
22: -82.3870697, 48.5418396, -82.3058319, 48.4317627, -130.8188324, 130.8476715
23: -55.1820908, 35.0601578, -55.0725174, 35.0145416, -90.1966324, 90.1326675
24: -64.8812103, 34.9667206, -64.8001556, 34.8380356, -99.7192383, 99.7668762
25: -60.3629379, 39.9742584, -60.2910500, 39.8941498, -100.2570877, 100.2653046
26: -93.2943573, 51.4981461, -93.1198349, 51.4356461, -144.7300110, 144.6179810
27: -68.8266983, 44.5310707, -68.7291794, 44.4483490, -113.2750473, 113.2602539
28: -56.8567734, 36.7264023, -56.7726097, 36.6852646, -93.5420227, 93.4990082
29: -81.8799133, 54.7719879, -81.7888412, 54.7178345, -136.5977478, 136.5608215
30: -68.3351746, 37.4815063, -68.2281342, 37.4193954, -105.7545471, 105.7096329
31: -63.1792641, 30.9327888, -63.0296745, 30.8910427, -94.0703049, 93.9624634
32: -65.9909515, 48.4695549, -65.7688293, 48.4345093, -114.4254608, 114.2383804
33: -100.6110687, 58.8378105, -100.5107956, 58.6705093, -159.2815704, 159.3486023
34: -85.4994736, 44.8283195, -85.4329453, 44.7175140, -130.2169800, 130.2612610
35: -81.3249817, 47.6955948, -81.2469635, 47.5644035, -128.8893890, 128.9425354
36: -82.9250031, 48.6808319, -82.8357315, 48.6259689, -131.5509644, 131.5165710
37: -115.8588867, 48.4014053, -115.7355804, 48.3299065, -164.1887970, 164.1369934
38: -102.6719284, 63.9023438, -102.5466309, 63.8297958, -166.5017090, 166.4489746
39: -123.0471649, 55.0478668, -122.8978195, 54.9467888, -177.9939575, 177.9456787
40: -97.3166962, 47.7300110, -97.2038040, 47.6545258, -144.9712219, 144.9338074
41: -67.4609299, 40.3436050, -67.3315582, 40.2843018, -107.7452316, 107.6751633
42: -50.0483093, 45.4863853, -49.8649788, 45.4305534, -95.4788589, 95.3513641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=371, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

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
type: A, layer: 1, pos: 1671
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8897529
time: 63.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189
time: 85.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 151.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.9444702
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.7831248
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8897529
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 151.46
Output dim: 2, lower bound: -53.8484139, upper bound: 53.8312189

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.2974510, 45.8054733, -62.4894409, 45.8760223, -108.1734772, 108.2949066
1: -39.6105385, 41.6874847, -39.7532196, 41.7812271, -81.3917618, 81.4407043
2: -36.6650162, 43.6796570, -36.8638153, 43.7284889, -80.3935089, 80.5434723
3: -44.6159592, 51.7459412, -44.8272934, 51.8564911, -96.4724503, 96.5732346
4: -52.3005066, 40.4381142, -52.4298782, 40.4672127, -92.7677155, 92.8679962
5: -46.4181786, 56.7375145, -46.6563263, 56.8427658, -103.2609329, 103.3938446
6: -67.5828400, 41.3088951, -67.7138367, 41.5260010, -109.1088257, 109.0227356
7: -56.6835175, 52.6792984, -56.9377518, 52.8617706, -109.5452805, 109.6170502
8: -46.9002724, 46.9247665, -47.1089859, 47.0203133, -93.9205780, 94.0337524
9: -49.0770721, 52.1572609, -49.2260208, 52.4339485, -101.5110168, 101.3832855
10: -78.5927734, 75.8385468, -78.7142563, 76.2485046, -154.8412781, 154.5527954
11: -79.9730606, 52.5068817, -79.9942474, 52.7482796, -132.7213440, 132.5011292
12: -73.8407593, 57.7721100, -74.0144119, 58.4132729, -132.2540283, 131.7865143
13: -70.5163879, 65.9553833, -70.6942444, 66.3209381, -136.8373260, 136.6496124
14: -106.0097885, 56.3681717, -106.3014908, 56.9135208, -162.9233093, 162.6696625
15: -58.5619392, 50.3047485, -58.7805557, 50.4713669, -109.0333099, 109.0853043
16: -82.5627289, 65.9517975, -82.6750183, 66.1345444, -148.6972656, 148.6268158
17: -118.3807144, 77.7334137, -118.6017151, 78.3770447, -196.7577362, 196.3351288
18: -68.8300476, 41.9069901, -68.9997253, 41.9929047, -110.8229523, 110.9066925
19: -59.8491249, 24.8519554, -59.9262123, 24.8876190, -84.7367401, 84.7781677
20: -53.9423065, 32.1674728, -54.0179443, 32.2723541, -86.2146606, 86.1854172
21: -72.1587830, 36.4361458, -72.2110291, 36.5797577, -108.7385406, 108.6471710
22: -81.4692535, 47.6553574, -81.7641373, 47.8988762, -129.3681183, 129.4194794
23: -54.6355095, 34.6117401, -54.7615280, 34.6469193, -89.2824249, 89.3732605
24: -64.0630188, 34.5814247, -64.3079224, 34.6670990, -98.7301025, 98.8893433
25: -59.8052483, 39.4201469, -59.9705811, 39.5480881, -99.3533325, 99.3907318
26: -92.0956345, 49.9328499, -92.3515015, 50.3938446, -142.4894714, 142.2843475
27: -67.8153152, 44.1787338, -68.1028824, 44.2595596, -112.0748749, 112.2816162
28: -56.3428993, 36.4446487, -56.4801178, 36.4772034, -92.8200989, 92.9247665
29: -81.1684189, 53.7364578, -81.3757019, 54.0464859, -135.2149048, 135.1121521
30: -67.7453308, 36.7138443, -67.9159088, 36.8404922, -104.5858231, 104.6297531
31: -62.3513184, 30.5607586, -62.5133743, 30.6137695, -92.9650879, 93.0741348
32: -65.3331909, 47.6735535, -65.4378204, 47.8656235, -113.1988068, 113.1113739
33: -99.3599319, 58.2915535, -99.6604004, 58.2790031, -157.6389160, 157.9519501
34: -84.6232224, 44.3188286, -84.8801422, 44.3803596, -129.0035858, 129.1989746
35: -80.2630386, 47.2186432, -80.5085831, 47.2423935, -127.5054321, 127.7272263
36: -82.2563629, 48.3115540, -82.3925171, 48.3659363, -130.6222992, 130.7040710
37: -114.9489746, 47.9232483, -115.1961746, 47.9942398, -162.9432068, 163.1194153
38: -101.6793442, 63.3336639, -101.9307022, 63.4727135, -165.1520538, 165.2643738
39: -121.9983368, 54.5634842, -122.2197037, 54.6698456, -176.6681824, 176.7831879
40: -96.2636948, 47.2583275, -96.5623550, 47.3690529, -143.6327515, 143.8206787
41: -66.7709427, 39.6785278, -66.9654312, 39.8237419, -106.5946808, 106.6439590
42: -49.4979553, 44.3442497, -49.5787010, 44.5081749, -94.0061340, 93.9229431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 982
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
type: B, layer: 1, pos: 1703
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
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1631
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
time: 71.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7766046
time: 60.19 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -62.6726723, 46.0850449, -62.5403252, 45.9820938, -108.6547699, 108.6253662
1: -39.8636398, 41.8633881, -39.7829056, 41.8242035, -81.6878433, 81.6462936
2: -37.0097198, 43.9358101, -36.8916168, 43.8339539, -80.8436737, 80.8274231
3: -45.0056877, 52.0655251, -44.8600235, 51.9740715, -96.9797592, 96.9255524
4: -52.5704117, 40.5937691, -52.4581337, 40.5112190, -93.0816345, 93.0518951
5: -46.8382111, 57.1032829, -46.6951637, 56.9908371, -103.8290253, 103.7984467
6: -67.8188477, 41.5784607, -67.7573700, 41.5871162, -109.4059601, 109.3358307
7: -57.1298447, 53.0021973, -56.9889793, 52.9732399, -110.1030884, 109.9911728
8: -47.2276230, 47.2081337, -47.1304245, 47.1201057, -94.3477097, 94.3385544
9: -49.4203720, 52.5572968, -49.3476944, 52.4642181, -101.8845901, 101.9049911
10: -79.1572342, 76.5302353, -78.9308014, 76.2908020, -155.4480286, 155.4610291
11: -80.1499481, 52.8639297, -80.0692139, 52.7695427, -132.9194946, 132.9331360
12: -74.5541534, 58.6205826, -74.3218765, 58.4462204, -133.0003662, 132.9424438
13: -70.8605957, 66.3656921, -70.8088531, 66.3675385, -137.2281342, 137.1745300
14: -106.7925797, 57.0644760, -106.5944443, 56.9345779, -163.7271576, 163.6589203
15: -58.8776436, 50.5770454, -58.8429337, 50.5182648, -109.3959045, 109.4199753
16: -82.8301010, 66.3036118, -82.7554321, 66.1879959, -149.0180969, 149.0590210
17: -119.0269623, 78.4792557, -118.8594589, 78.3916321, -197.4185791, 197.3387146
18: -69.1606750, 42.1450882, -69.0861816, 42.0301857, -111.1908569, 111.2312698
19: -60.0611725, 24.9691257, -59.9861832, 24.9037056, -84.9648743, 84.9553070
20: -54.1688347, 32.3319969, -54.0833015, 32.2900391, -86.4588699, 86.4152985
21: -72.4097748, 36.7013931, -72.2870941, 36.6052017, -109.0149765, 108.9884796
22: -81.9124374, 48.0475426, -81.9178696, 47.9330940, -129.8455353, 129.9654083
23: -54.8753357, 34.7375717, -54.8177795, 34.6667519, -89.5420837, 89.5553436
24: -64.3410339, 34.7120628, -64.3489685, 34.7019653, -99.0429993, 99.0610352
25: -60.0170975, 39.6554260, -60.0266342, 39.5868225, -99.6039200, 99.6820602
26: -92.7913742, 50.6541290, -92.6226883, 50.4374542, -143.2288208, 143.2768250
27: -68.1732407, 44.3218842, -68.1439209, 44.3016434, -112.4748688, 112.4658051
28: -56.5752945, 36.5423317, -56.5282364, 36.5040588, -93.0793533, 93.0705719
29: -81.5319214, 54.1880417, -81.5011215, 54.0724831, -135.6044006, 135.6891632
30: -68.0074234, 36.9410210, -67.9635162, 36.8659325, -104.8733521, 104.9045410
31: -62.6406975, 30.7029114, -62.5700989, 30.6428089, -93.2835083, 93.2730103
32: -65.5335236, 47.9033165, -65.4859772, 47.8931122, -113.4266357, 113.3892975
33: -99.7807617, 58.4483910, -99.7059860, 58.3342209, -158.1149750, 158.1543732
34: -85.0025330, 44.5243301, -84.9232483, 44.4474258, -129.4499512, 129.4475708
35: -80.6374969, 47.3878822, -80.5441895, 47.2961960, -127.9336929, 127.9320602
36: -82.5322266, 48.4417496, -82.4416275, 48.4068108, -130.9390259, 130.8833771
37: -115.2912292, 48.1101990, -115.2568665, 48.0388412, -163.3300476, 163.3670654
38: -102.1019974, 63.5702629, -101.9788055, 63.5381546, -165.6401520, 165.5490723
39: -122.3194427, 54.7892609, -122.2645035, 54.7296143, -177.0490570, 177.0537567
40: -96.6688766, 47.5320244, -96.6023102, 47.4819260, -144.1508026, 144.1343384
41: -67.0522614, 39.9075737, -67.0012207, 39.8808899, -106.9331512, 106.9087982
42: -49.6664848, 44.6320724, -49.6231384, 44.5468063, -94.2132874, 94.2552109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 679
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
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1641
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
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1631
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
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1569

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
time: 69.81 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
time: 72.01 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5803375, 45.8684006, -63.0774002, 46.0987244, -108.6790619, 108.9458008
1: -39.7811394, 41.7404938, -40.1029663, 41.9197235, -81.7008514, 81.8434448
2: -36.9603882, 43.7193565, -37.4630241, 43.9075050, -80.8678894, 81.1823807
3: -44.9207993, 51.8091278, -45.4459419, 52.0853081, -97.0061035, 97.2550583
4: -52.6306839, 40.4964256, -53.1012650, 40.6539192, -93.2845993, 93.5976791
5: -46.7252617, 56.7939682, -47.2725754, 57.0918388, -103.8171005, 104.0665359
6: -67.6848755, 41.4732704, -67.9475327, 41.8740120, -109.5588760, 109.4207916
7: -56.9292107, 52.7347107, -57.4598694, 53.0223160, -109.9515228, 110.1945801
8: -47.1929550, 46.9812546, -47.7045822, 47.2118759, -94.4048309, 94.6858368
9: -49.1753502, 52.4050636, -49.4692268, 52.9458847, -102.1212311, 101.8742905
10: -78.7211990, 76.4079666, -79.1633911, 77.3934937, -156.1146851, 155.5713501
11: -80.0660629, 52.9274216, -80.2550201, 53.5809784, -133.6470337, 133.1824341
12: -73.9110107, 58.3810081, -74.3892822, 59.6442719, -133.5552673, 132.7702942
13: -70.6106262, 66.1085052, -70.8999252, 66.6720734, -137.2826996, 137.0084229
14: -106.1640625, 56.7433090, -106.7738495, 57.6791000, -163.8431702, 163.5171509
15: -58.8142052, 50.4046555, -59.3129883, 50.7041893, -109.5183868, 109.7176361
16: -82.6993332, 66.2538910, -82.9970398, 66.7652969, -149.4646301, 149.2509308
17: -118.4831696, 78.2634888, -118.9634705, 79.4484100, -197.9315796, 197.2269592
18: -68.9505234, 42.1019783, -69.3235779, 42.3947067, -111.3452301, 111.4255447
19: -59.9318733, 24.9882431, -60.1551323, 25.1571007, -85.0889740, 85.1433640
20: -54.0285454, 32.3048744, -54.2531052, 32.5519562, -86.5805054, 86.5579834
21: -72.2470093, 36.6616745, -72.4965820, 37.0383224, -109.2853165, 109.1582489
22: -81.5854111, 47.8500099, -82.0377502, 48.3049316, -129.8903503, 129.8877411
23: -54.7113800, 34.7488441, -54.9591904, 34.9287758, -89.6401367, 89.7080307
24: -64.2014771, 34.6336899, -64.6109467, 34.7688637, -98.9703369, 99.2446365
25: -59.8798981, 39.5380478, -60.1473007, 39.7980270, -99.6779251, 99.6853485
26: -92.1973038, 50.3365440, -92.7355347, 51.2213287, -143.4186249, 143.0720825
27: -68.0090790, 44.2283707, -68.5192719, 44.3706741, -112.3797531, 112.7476425
28: -56.4203949, 36.5100403, -56.6695137, 36.6214638, -93.0418549, 93.1795502
29: -81.2599564, 53.9959831, -81.5929871, 54.5661888, -135.8261414, 135.5889740
30: -67.8247070, 36.9216270, -68.1150513, 37.2666817, -105.0913773, 105.0366821
31: -62.4856567, 30.6624336, -62.8552856, 30.8205299, -93.3061829, 93.5177155
32: -65.4331207, 47.8619156, -65.6564484, 48.2487373, -113.6818542, 113.5183640
33: -99.6526260, 58.3833580, -100.2606888, 58.5536575, -158.2062683, 158.6440430
34: -84.8139191, 44.3995438, -85.2772598, 44.5976639, -129.4115906, 129.6767883
35: -80.5327606, 47.2925262, -81.0519257, 47.4611244, -127.9938812, 128.3444519
36: -82.4086609, 48.3789330, -82.7192230, 48.5324860, -130.9411469, 131.0981598
37: -115.1083527, 48.0275497, -115.5444336, 48.2294998, -163.3378448, 163.5719910
38: -101.9003754, 63.4233704, -102.3955994, 63.6858330, -165.5861816, 165.8189697
39: -122.2157669, 54.6306381, -122.6805801, 54.8449402, -177.0606995, 177.3112183
40: -96.4742432, 47.3024635, -97.0132751, 47.5068207, -143.9810486, 144.3157349
41: -66.8844986, 39.8130455, -67.2182922, 40.1090660, -106.9935608, 107.0313416
42: -49.5800972, 44.6707344, -49.7657318, 45.1725197, -94.7526169, 94.4364624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8119656
time: 72.49 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
time: 68.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -62.9571381, 46.1476517, -63.1283226, 46.2046394, -109.1617737, 109.2759705
1: -40.0350113, 41.9165802, -40.1324921, 41.9634209, -81.9984283, 82.0490723
2: -37.3067169, 43.9753227, -37.4912033, 44.0101166, -81.3168335, 81.4665222
3: -45.3128357, 52.1284447, -45.4789162, 52.2023468, -97.5151825, 97.6073608
4: -52.9022293, 40.6524086, -53.1294556, 40.6981392, -93.6003647, 93.7818604
5: -47.1413116, 57.1598320, -47.3115997, 57.2335358, -104.3748474, 104.4714279
6: -67.9215851, 41.7464638, -67.9913330, 41.9376755, -109.8592529, 109.7377930
7: -57.3792076, 53.0568848, -57.5111771, 53.1344452, -110.5136490, 110.5680618
8: -47.5213814, 47.2645035, -47.7263222, 47.3115387, -94.8329086, 94.9908295
9: -49.5186844, 52.8062286, -49.5909615, 52.9762192, -102.4949036, 102.3971786
10: -79.2852631, 77.1019363, -79.3786621, 77.4353333, -156.7205811, 156.4805908
11: -80.2458649, 53.2871399, -80.3307800, 53.6031609, -133.8490143, 133.6179199
12: -74.6240845, 59.2323685, -74.6963120, 59.6777725, -134.3018494, 133.9286804
13: -70.9549942, 66.5205383, -71.0147400, 66.7195435, -137.6745300, 137.5352631
14: -106.9465179, 57.4405289, -107.0675888, 57.6999550, -164.6464691, 164.5081177
15: -59.1270332, 50.6781578, -59.3678131, 50.7512665, -109.8782959, 110.0459595
16: -82.9684448, 66.6104279, -83.0780792, 66.8177795, -149.7862244, 149.6885071
17: -119.1293564, 79.0125580, -119.2216263, 79.4641342, -198.5934448, 198.2341766
18: -69.2824936, 42.3400841, -69.4106903, 42.4319801, -111.7144547, 111.7507782
19: -60.1443787, 25.1042080, -60.2154617, 25.1734619, -85.3178329, 85.3196640
20: -54.2552872, 32.4704742, -54.3194771, 32.5696945, -86.8249741, 86.7899475
21: -72.4981537, 36.9280624, -72.5734940, 37.0640335, -109.5621872, 109.5015564
22: -82.0288620, 48.2425003, -82.1905975, 48.3390388, -130.3679047, 130.4331055
23: -54.9522476, 34.8753815, -55.0158501, 34.9485054, -89.9007492, 89.8912354
24: -64.4812775, 34.7648430, -64.6509323, 34.8037415, -99.2850189, 99.4157715
25: -60.0925255, 39.7742462, -60.2037086, 39.8364868, -99.9290161, 99.9779510
26: -92.8920441, 51.0613365, -93.0077057, 51.2646523, -144.1566772, 144.0690460
27: -68.3686218, 44.3717499, -68.5601273, 44.4121628, -112.7807846, 112.9318771
28: -56.6537361, 36.6077194, -56.7177544, 36.6482086, -93.3019409, 93.3254700
29: -81.6230240, 54.4488754, -81.7185516, 54.5918579, -136.2148743, 136.1674194
30: -68.0884705, 37.1498260, -68.1631927, 37.2924232, -105.3808899, 105.3130188
31: -62.7795448, 30.8052292, -62.9114494, 30.8495941, -93.6291351, 93.7166595
32: -65.6342773, 48.0934296, -65.7049713, 48.2767181, -113.9109955, 113.7984009
33: -100.0752487, 58.5403442, -100.3064728, 58.6100235, -158.6852722, 158.8468170
34: -85.1946716, 44.6049080, -85.3203354, 44.6652184, -129.8598938, 129.9252472
35: -80.9093246, 47.4612274, -81.0877762, 47.5157967, -128.4251251, 128.5489960
36: -82.6879730, 48.5109062, -82.7683487, 48.5739288, -131.2619019, 131.2792511
37: -115.4521484, 48.2150230, -115.6052551, 48.2746429, -163.7267914, 163.8202820
38: -102.3251190, 63.6608238, -102.4437103, 63.7506943, -166.0758057, 166.1045380
39: -122.5378342, 54.8571205, -122.7249680, 54.9052238, -177.4430237, 177.5820923
40: -96.8805084, 47.5765305, -97.0530472, 47.6204185, -144.5009308, 144.6295776
41: -67.1673889, 40.0371666, -67.2542343, 40.1591797, -107.3265686, 107.2913971
42: -49.7499352, 44.9629784, -49.8103600, 45.2112274, -94.9611664, 94.7733383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
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
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 674
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
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8678289
time: 70.65 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
time: 72.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -62.6876068, 46.0393181, -62.6269684, 45.9138222, -108.6014175, 108.6662827
1: -39.8298111, 41.7998924, -39.8277626, 41.8097038, -81.6395111, 81.6276550
2: -37.0373077, 43.9190063, -37.0147781, 43.7504082, -80.7877045, 80.9337845
3: -44.9688339, 51.9648590, -44.9646339, 51.8962173, -96.8650513, 96.9294739
4: -52.7640114, 40.7276382, -52.6146240, 40.4996185, -93.2636185, 93.3422623
5: -46.7802353, 57.0470009, -46.8004837, 56.8817940, -103.6620331, 103.8474808
6: -67.8796768, 41.6270218, -67.7664566, 41.6443100, -109.5239792, 109.3934784
7: -57.0009422, 52.8069725, -57.0416908, 52.8947525, -109.8956909, 109.8486557
8: -47.3218918, 47.1947289, -47.2759132, 47.0528793, -94.3747711, 94.4706421
9: -49.3078880, 52.5326424, -49.2764473, 52.5741310, -101.8820038, 101.8090897
10: -79.0770035, 76.5235138, -78.7817841, 76.5273590, -155.6043549, 155.3052979
11: -80.4193268, 53.1122894, -80.0570679, 53.0046425, -133.4239502, 133.1693420
12: -74.4329376, 58.6434593, -74.0604095, 58.7855949, -133.2185364, 132.7038727
13: -70.7284088, 66.4006348, -70.7392807, 66.4748764, -137.2032776, 137.1399078
14: -106.4788742, 56.9001389, -106.3822708, 57.1428146, -163.6216583, 163.2824097
15: -58.9786453, 50.6419601, -58.9237213, 50.5284767, -109.5071259, 109.5656815
16: -82.9524612, 66.3538818, -82.7602081, 66.2898560, -149.2423096, 149.1140900
17: -118.9022598, 78.5892487, -118.6614304, 78.7447510, -197.6470032, 197.2506714
18: -69.1612701, 42.1198807, -69.0748596, 42.0600700, -111.2213440, 111.1947327
19: -60.0967026, 25.0055141, -59.9768524, 24.9423733, -85.0390778, 84.9823608
20: -54.1957245, 32.3702583, -54.0661812, 32.3517990, -86.5475159, 86.4364395
21: -72.5035324, 36.7164612, -72.2651978, 36.6927872, -109.1963196, 108.9816513
22: -81.7133484, 47.9394989, -81.8252945, 47.9815369, -129.6948853, 129.7648010
23: -54.8496361, 34.7803078, -54.8098145, 34.7044525, -89.5540924, 89.5901184
24: -64.4377136, 34.7699432, -64.4431305, 34.6937408, -99.1314545, 99.2130661
25: -60.0215950, 39.6065369, -60.0319633, 39.5978432, -99.6194305, 99.6384964
26: -92.4113617, 50.3566551, -92.4156570, 50.5524979, -142.9638672, 142.7723083
27: -68.2396622, 44.3160744, -68.2537994, 44.2849312, -112.5245972, 112.5698700
28: -56.5300064, 36.5495071, -56.5265045, 36.5069275, -93.0369339, 93.0760117
29: -81.3755798, 54.0439224, -81.4226456, 54.1630325, -135.5386047, 135.4665527
30: -67.9686737, 37.0194244, -67.9694748, 36.9539337, -104.9226074, 104.9888916
31: -62.7149696, 30.6705265, -62.6175766, 30.6455841, -93.3605499, 93.2881012
32: -65.6804047, 48.0240707, -65.4906769, 48.0102005, -113.6905975, 113.5147476
33: -99.8635788, 58.5710144, -99.8481216, 58.3288460, -158.1924133, 158.4191284
34: -84.9025574, 44.5184555, -84.9797592, 44.4200974, -129.3226624, 129.4981995
35: -80.6447449, 47.4372215, -80.6503677, 47.2821007, -127.9268494, 128.0875854
36: -82.4673157, 48.4682083, -82.4468079, 48.4108238, -130.8781433, 130.9150085
37: -115.3158188, 48.0931396, -115.3059616, 48.0405540, -163.3563538, 163.3991089
38: -101.9993439, 63.5498962, -102.0184250, 63.5386391, -165.5379639, 165.5683289
39: -122.4575653, 54.7372856, -122.3664093, 54.7017097, -177.1592712, 177.1036987
40: -96.6747742, 47.3854446, -96.6990738, 47.3901978, -144.0649719, 144.0845184
41: -67.0445175, 39.9321136, -67.0339050, 39.9179306, -106.9624481, 106.9660110
42: -49.7815781, 44.8204994, -49.6239433, 44.7047653, -94.4863434, 94.4444427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 680
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
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
time: 94.01 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
time: 79.22 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -63.0592651, 46.3166161, -62.6777153, 46.0198822, -109.0791473, 108.9943314
1: -40.0813522, 41.9757996, -39.8573151, 41.8528442, -81.9341812, 81.8331070
2: -37.3805084, 44.1741447, -37.0424347, 43.8557587, -81.2362671, 81.2165833
3: -45.3505745, 52.2792015, -44.9974327, 52.0132942, -97.3638611, 97.2766342
4: -53.0314827, 40.8821220, -52.6422462, 40.5436783, -93.5751648, 93.5243683
5: -47.1940613, 57.4077492, -46.8392792, 57.0297852, -104.2238464, 104.2470245
6: -68.1114273, 41.8894386, -67.8100815, 41.7045860, -109.8159943, 109.6995163
7: -57.4408188, 53.1310921, -57.0930252, 53.0068207, -110.4476395, 110.2241211
8: -47.6466370, 47.4764099, -47.2967415, 47.1525345, -94.7991638, 94.7731476
9: -49.6495667, 52.9301567, -49.3980865, 52.6039162, -102.2534790, 102.3282394
10: -79.6385880, 77.2114182, -78.9979858, 76.5690002, -156.2075806, 156.2094116
11: -80.5717621, 53.4680405, -80.1322250, 53.0256653, -133.5974274, 133.6002655
12: -75.1430969, 59.4888229, -74.3677292, 58.8179626, -133.9610596, 133.8565521
13: -71.0707245, 66.8069305, -70.8538742, 66.5206604, -137.5913849, 137.6607971
14: -107.2593002, 57.5944595, -106.6750641, 57.1635361, -164.4228210, 164.2695312
15: -59.2597084, 50.9098778, -58.9716949, 50.5755768, -109.8352814, 109.8815689
16: -83.2077026, 66.7116013, -82.8408508, 66.3416138, -149.5493011, 149.5524597
17: -119.5467911, 79.3297501, -118.9189148, 78.7581787, -198.3049622, 198.2486572
18: -69.4888306, 42.3575516, -69.1606140, 42.0975266, -111.5863495, 111.5181580
19: -60.3080254, 25.1211128, -60.0366821, 24.9584999, -85.2665100, 85.1577911
20: -54.4199066, 32.5343781, -54.1316872, 32.3694382, -86.7893448, 86.6660614
21: -72.7528229, 36.9811134, -72.3413086, 36.7182159, -109.4710388, 109.3224182
22: -82.1641006, 48.3275871, -81.9783020, 48.0161057, -130.1802063, 130.3058929
23: -55.0885658, 34.9051666, -54.8661613, 34.7241631, -89.8127136, 89.7713318
24: -64.7134247, 34.8995361, -64.4832993, 34.7287140, -99.4421387, 99.3828354
25: -60.2343102, 39.8405418, -60.0878792, 39.6365891, -99.8708954, 99.9284210
26: -93.1018372, 51.0664597, -92.6856079, 50.5963020, -143.6981201, 143.7520599
27: -68.5939713, 44.4609070, -68.2938461, 44.3269043, -112.9208755, 112.7547531
28: -56.7619324, 36.6461067, -56.5746803, 36.5338135, -93.2957458, 93.2207870
29: -81.7429810, 54.4928513, -81.5473175, 54.1891212, -135.9320984, 136.0401611
30: -68.2302704, 37.2449493, -68.0170746, 36.9791489, -105.2094193, 105.2620087
31: -63.0112190, 30.8117218, -62.6731987, 30.6747875, -93.6860046, 93.4849243
32: -65.8694305, 48.2519684, -65.5389328, 48.0373802, -113.9067917, 113.7908936
33: -100.2826920, 58.7239647, -99.8934784, 58.3842087, -158.6668701, 158.6174469
34: -85.2814331, 44.7226830, -85.0227509, 44.4871445, -129.7685699, 129.7454376
35: -81.0178528, 47.6041412, -80.6858215, 47.3360176, -128.3538513, 128.2899628
36: -82.7433548, 48.5962105, -82.4958954, 48.4518242, -131.1951752, 131.0921021
37: -115.6557541, 48.2786789, -115.3664627, 48.0852966, -163.7410583, 163.6451416
38: -102.4179382, 63.7862282, -102.0663757, 63.6037903, -166.0217285, 165.8526001
39: -122.7727356, 54.9611626, -122.4094772, 54.7616501, -177.5343933, 177.3706360
40: -97.0766907, 47.6595650, -96.7387466, 47.5032234, -144.5798950, 144.3983154
41: -67.3277435, 40.1484070, -67.0697479, 39.9700241, -107.2977676, 107.2181549
42: -49.9466476, 45.1076660, -49.6684570, 44.7426491, -94.6893005, 94.7761230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1767
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
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
time: 77.09 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
time: 69.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -62.9705658, 46.1015358, -63.2152367, 46.1359329, -109.1064911, 109.3167648
1: -40.0001335, 41.8528519, -40.1773376, 41.9479828, -81.9481125, 82.0301895
2: -37.3325768, 43.9583588, -37.6139755, 43.9291191, -81.2616959, 81.5723267
3: -45.2737808, 52.0274849, -45.5834274, 52.1242332, -97.3980026, 97.6109161
4: -53.0939369, 40.7856178, -53.2859077, 40.6861153, -93.7800522, 94.0715256
5: -47.0873108, 57.1031532, -47.4168587, 57.1302910, -104.2176056, 104.5200119
6: -67.9811401, 41.7945976, -67.9998550, 41.9927292, -109.9738617, 109.7944489
7: -57.2466698, 52.8624420, -57.5638924, 53.0547600, -110.3014297, 110.4263306
8: -47.6144943, 47.2507706, -47.8715973, 47.2441063, -94.8585815, 95.1223679
9: -49.4048004, 52.7806244, -49.5187263, 53.0863533, -102.4911499, 102.2993469
10: -79.2040482, 77.0926819, -79.2299728, 77.6721420, -156.8761902, 156.3226624
11: -80.5098877, 53.5328484, -80.3170776, 53.8374214, -134.3473053, 133.8499298
12: -74.5026093, 59.2524796, -74.4348068, 60.0167046, -134.5193176, 133.6872864
13: -70.8218079, 66.5550842, -70.9445343, 66.8259430, -137.6477509, 137.4996033
14: -106.6322327, 57.2759972, -106.8537674, 57.9088554, -164.5410767, 164.1297607
15: -59.2425690, 50.7413597, -59.4714355, 50.7607307, -110.0032959, 110.2127914
16: -83.0879440, 66.6628113, -83.0805817, 66.9265289, -150.0144653, 149.7433929
17: -119.0041428, 79.1201553, -119.0226440, 79.8169403, -198.8210754, 198.1427917
18: -69.2826538, 42.3157425, -69.3983765, 42.4633789, -111.7460327, 111.7141190
19: -60.1798439, 25.1415787, -60.2050247, 25.2117538, -85.3915863, 85.3466034
20: -54.2814331, 32.5077820, -54.3006096, 32.6314697, -86.9129028, 86.8083801
21: -72.5910950, 36.9417496, -72.5500717, 37.1511459, -109.7422409, 109.4918213
22: -81.8297729, 48.1330261, -82.0984039, 48.3872147, -130.2169800, 130.2314301
23: -54.9252434, 34.9171791, -55.0071564, 34.9861374, -89.9113770, 89.9243317
24: -64.5770187, 34.8214684, -64.7475433, 34.7952766, -99.3722992, 99.5690079
25: -60.0967331, 39.7236710, -60.2085304, 39.8476410, -99.9443741, 99.9321976
26: -92.5129776, 50.7601318, -92.7989502, 51.3799400, -143.8929138, 143.5590820
27: -68.4342957, 44.3651657, -68.6711273, 44.3959007, -112.8302002, 113.0362854
28: -56.6077385, 36.6147423, -56.7160072, 36.6511154, -93.2588501, 93.3307343
29: -81.4673615, 54.3029823, -81.6401520, 54.6826706, -136.1500244, 135.9431305
30: -68.0479584, 37.2273865, -68.1682281, 37.3802223, -105.4281616, 105.3956070
31: -62.8530922, 30.7720089, -62.9603920, 30.8522072, -93.7052994, 93.7323990
32: -65.7787170, 48.2126160, -65.7089005, 48.3935776, -114.1722946, 113.9215164
33: -100.1566772, 58.6621742, -100.4486313, 58.6025581, -158.7592163, 159.1108093
34: -85.0938492, 44.5988464, -85.3772507, 44.6368713, -129.7307129, 129.9760895
35: -80.9148865, 47.5102997, -81.1938782, 47.5001068, -128.4149933, 128.7041779
36: -82.6216278, 48.5361252, -82.7734985, 48.5768204, -131.1984558, 131.3096313
37: -115.4767303, 48.1971283, -115.6543274, 48.2754288, -163.7521667, 163.8514404
38: -102.2209854, 63.6405830, -102.4837418, 63.7527962, -165.9737854, 166.1243286
39: -122.6762848, 54.8043938, -122.8276367, 54.8764343, -177.5527039, 177.6320343
40: -96.8858337, 47.4297447, -97.1504593, 47.5274010, -144.4132385, 144.5802002
41: -67.1576843, 40.0675659, -67.2863007, 40.2043457, -107.3620300, 107.3538589
42: -49.8623276, 45.1483688, -49.8106270, 45.3693619, -95.2316895, 94.9589996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1400
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
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1018
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
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 629
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 613
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
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1361
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
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1323
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
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8119656
time: 66.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.9423880, upper bound: 53.8876334
time: 75.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -63.3437233, 46.3784790, -63.2659836, 46.2418060, -109.5855255, 109.6444626
1: -40.2524147, 42.0289764, -40.2066879, 41.9916039, -82.2440033, 82.2356644
2: -37.6773300, 44.2132835, -37.6419373, 44.0316238, -81.7089539, 81.8552246
3: -45.6578331, 52.3415298, -45.6164207, 52.2408218, -97.8986511, 97.9579391
4: -53.3629684, 40.9404106, -53.3134537, 40.7303734, -94.0933380, 94.2538605
5: -47.4970627, 57.4637108, -47.4557991, 57.2720070, -104.7690506, 104.9195099
6: -68.2134399, 42.0616875, -68.0435638, 42.0558281, -110.2692719, 110.1052399
7: -57.6907997, 53.1858978, -57.6150932, 53.1673584, -110.8581543, 110.8009949
8: -47.9402924, 47.5322723, -47.8926811, 47.3436241, -95.2839203, 95.4249496
9: -49.7465591, 53.1792603, -49.6403580, 53.1162338, -102.8627777, 102.8196182
10: -79.7652054, 77.7827072, -79.4449463, 77.7133026, -157.4785156, 157.2276611
11: -80.6659317, 53.8910751, -80.3930359, 53.8592567, -134.5251923, 134.2841187
12: -75.2123871, 60.1005630, -74.7416229, 60.0496063, -135.2619934, 134.8421936
13: -71.1642151, 66.9629822, -71.0593414, 66.8724823, -138.0366821, 138.0223236
14: -107.4123230, 57.9712410, -107.1473770, 57.9293213, -165.3416290, 165.1186218
15: -59.5148926, 51.0102844, -59.5022202, 50.8080254, -110.3229141, 110.5125046
16: -83.3449249, 67.0238037, -83.1616898, 66.9775620, -150.3224792, 150.1854858
17: -119.6484909, 79.8638763, -119.2804871, 79.8312378, -199.4797058, 199.1443634
18: -69.6116028, 42.5537300, -69.4848328, 42.5008087, -112.1124115, 112.0385590
19: -60.3915215, 25.2558556, -60.2652435, 25.2281475, -85.6196671, 85.5211029
20: -54.5056076, 32.6728477, -54.3671074, 32.6491394, -87.1547470, 87.0399551
21: -72.8404236, 37.2074509, -72.6270447, 37.1768074, -110.0172272, 109.8344879
22: -82.2801514, 48.5213051, -82.2505951, 48.4216423, -130.7017975, 130.7718964
23: -55.1651611, 35.0427246, -55.0638924, 35.0057297, -90.1708908, 90.1066132
24: -64.8544235, 34.9515419, -64.7868958, 34.8302155, -99.6846390, 99.7384338
25: -60.3097534, 39.9584160, -60.2647934, 39.8861694, -100.1959229, 100.2232056
26: -93.2027206, 51.4737358, -93.0700226, 51.4233475, -144.6260681, 144.5437622
27: -68.7901230, 44.5101013, -68.7110519, 44.4371490, -113.2272720, 113.2211533
28: -56.8405495, 36.7114258, -56.7642365, 36.6778030, -93.5183487, 93.4756546
29: -81.8343964, 54.7532158, -81.7650223, 54.7085037, -136.5428772, 136.5182343
30: -68.3115387, 37.4538918, -68.2163849, 37.4057045, -105.7172394, 105.6702728
31: -63.1523323, 30.9137726, -63.0160751, 30.8813934, -94.0337219, 93.9298477
32: -65.9684219, 48.4422226, -65.7574921, 48.4211617, -114.3895798, 114.1996994
33: -100.5773621, 58.8152580, -100.4941711, 58.6591034, -159.2364502, 159.3094177
34: -85.4741440, 44.8028984, -85.4202194, 44.7043953, -130.1785431, 130.2231140
35: -81.2899170, 47.6766434, -81.2295532, 47.5548172, -128.8447266, 128.9061890
36: -82.8991089, 48.6656342, -82.8227158, 48.6183167, -131.5174255, 131.4883423
37: -115.8171387, 48.3832207, -115.7151260, 48.3207321, -164.1378632, 164.0983429
38: -102.6416168, 63.8777084, -102.5316925, 63.8174133, -166.4590149, 166.4093933
39: -122.9925613, 55.0289459, -122.8712234, 54.9369431, -177.9295044, 177.9001465
40: -97.2887192, 47.7042656, -97.1899261, 47.6410980, -144.9298096, 144.8941956
41: -67.4425507, 40.2818871, -67.3223419, 40.2528000, -107.6953430, 107.6042252
42: -50.0289917, 45.4389496, -49.8553314, 45.4073906, -95.4363861, 95.2942810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

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
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
time: 88.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.8330470, upper bound: 53.9423876
time: 73.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 164.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7766046
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8119656
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8678289
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8119656
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.9423880, upper bound: 53.8876334
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 164.24
Output dim: 2, lower bound: -53.8330470, upper bound: 53.9423876

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -62.0695000, 45.7421761, -62.0434303, 45.7519302, -107.8214264, 107.7855911
1: -39.4806290, 41.6324005, -39.5021286, 41.6725121, -81.1531372, 81.1345291
2: -36.4525871, 43.6356354, -36.4496651, 43.6426239, -80.0952148, 80.0852966
3: -44.3702736, 51.6827431, -44.3453903, 51.7314110, -96.1016846, 96.0281372
4: -52.0455742, 40.3820457, -51.9384003, 40.3575783, -92.4031525, 92.3204498
5: -46.1959229, 56.6790810, -46.2203369, 56.7280006, -102.9239197, 102.8994141
6: -67.4854431, 41.1927452, -67.5231018, 41.3019905, -108.7874298, 108.7158508
7: -56.5147057, 52.6266251, -56.6050644, 52.7583237, -109.2730255, 109.2316818
8: -46.7147255, 46.8612823, -46.7507591, 46.8959770, -93.6107025, 93.6120377
9: -48.9454079, 52.0248489, -48.9702377, 52.1818962, -101.1273041, 100.9950867
10: -78.4771500, 75.3692932, -78.4892883, 75.3382263, -153.8153687, 153.8585815
11: -79.8794556, 52.0781326, -79.8084717, 51.9187317, -131.7981873, 131.8865967
12: -73.7688904, 57.3489151, -73.8736420, 57.5820618, -131.3509521, 131.2225647
13: -70.3051376, 65.8266754, -70.2844391, 66.0677719, -136.3728943, 136.1111145
14: -105.8599777, 56.0255775, -106.0072403, 56.2369881, -162.0969696, 162.0328217
15: -58.3419418, 50.2098885, -58.3592644, 50.2839699, -108.6259155, 108.5691452
16: -82.4173813, 65.7202225, -82.3887482, 65.7027740, -148.1201477, 148.1089783
17: -118.2706909, 77.2094269, -118.3862228, 77.3434830, -195.6141663, 195.5956421
18: -68.7083282, 41.6711502, -68.7587433, 41.5286865, -110.2369919, 110.4298782
19: -59.7688026, 24.6699982, -59.7687111, 24.5320892, -84.3008881, 84.4387054
20: -53.8616028, 32.0198669, -53.8583679, 31.9816246, -85.8432159, 85.8782272
21: -72.0684814, 36.1730957, -72.0350189, 36.0610428, -108.1295242, 108.2081146
22: -81.3669434, 47.4561424, -81.5668259, 47.5071182, -128.8740540, 129.0229645
23: -54.5645981, 34.4269943, -54.6214485, 34.2862816, -88.8508759, 89.0484467
24: -63.9714432, 34.4869919, -64.1309738, 34.4779816, -98.4494247, 98.6179657
25: -59.7291832, 39.2816582, -59.8216705, 39.2770767, -99.0062485, 99.1033325
26: -91.9987640, 49.5973663, -92.1612396, 49.7275887, -141.7263489, 141.7586060
27: -67.6754837, 44.0701065, -67.8333893, 44.0514412, -111.7269287, 111.9034958
28: -56.2693176, 36.3507957, -56.3359451, 36.2959061, -92.5652237, 92.6867371
29: -81.0872345, 53.4638290, -81.2189178, 53.5106201, -134.5978546, 134.6827393
30: -67.6631012, 36.5080109, -67.7534180, 36.4398232, -104.1028976, 104.2614212
31: -62.2322807, 30.3862114, -62.2752151, 30.2721424, -92.5044174, 92.6614227
32: -65.2250671, 47.5568581, -65.2256622, 47.6425667, -112.8676147, 112.7825089
33: -99.0503693, 58.2127533, -99.0528030, 58.1239662, -157.1743164, 157.2655487
34: -84.4178314, 44.2434731, -84.4737930, 44.2342834, -128.6521149, 128.7172699
35: -79.9755707, 47.1521301, -79.9426575, 47.1139450, -127.0895081, 127.0947876
36: -82.0386353, 48.2521057, -81.9681854, 48.2478333, -130.2864532, 130.2202911
37: -114.7807388, 47.8293228, -114.8709641, 47.8121643, -162.5928955, 162.7002869
38: -101.4665222, 63.2510071, -101.5177994, 63.3119507, -164.7784729, 164.7687988
39: -121.7189941, 54.5029449, -121.6721344, 54.5515289, -176.2705078, 176.1750793
40: -96.0679169, 47.2174301, -96.1781845, 47.2876434, -143.3555603, 143.3956146
41: -66.6651917, 39.5681267, -66.7583160, 39.6133118, -106.2785034, 106.3264465
42: -49.4201813, 44.0885773, -49.4252739, 44.0126801, -93.4328537, 93.5138550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 664
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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1608
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
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 629
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
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1352
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
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1288

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
time: 73.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
time: 87.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -62.2704048, 45.7963409, -62.5390816, 46.0693130, -108.3397141, 108.3354187
1: -39.5940933, 41.6786575, -39.7732887, 41.8393288, -81.4334259, 81.4519501
2: -36.6437378, 43.6730804, -36.8660049, 43.8830185, -80.5267563, 80.5390778
3: -44.5891151, 51.7351875, -44.8202705, 52.0938797, -96.6829834, 96.5554504
4: -52.2722893, 40.4284058, -52.4368057, 40.5671730, -92.8394623, 92.8652039
5: -46.3939514, 56.7277679, -46.6579514, 57.0625038, -103.4564514, 103.3857193
6: -67.5669327, 41.2629738, -67.7511444, 41.5166130, -109.0835419, 109.0141144
7: -56.6623459, 52.6676369, -56.9627266, 52.9089317, -109.5712738, 109.6303635
8: -46.8782158, 46.9146690, -47.1173401, 47.1102486, -93.9884491, 94.0320129
9: -49.0498848, 52.1345177, -49.2488441, 52.4575615, -101.5074463, 101.3833618
10: -78.5780411, 75.7866058, -79.0480499, 76.2354507, -154.8134918, 154.8346558
11: -79.9596100, 52.4730225, -80.3306732, 52.7179108, -132.6775208, 132.8036957
12: -73.8278198, 57.7321930, -74.3811722, 58.4004211, -132.2282410, 132.1133728
13: -70.4822998, 65.9372253, -70.7078781, 66.5323868, -137.0146790, 136.6451111
14: -105.9877625, 56.3330994, -106.6299591, 56.8896217, -162.8773804, 162.9630585
15: -58.5107841, 50.2907829, -58.7912445, 50.5049210, -109.0157013, 109.0820084
16: -82.5427094, 65.9091797, -82.7666702, 66.1368103, -148.6795044, 148.6758423
17: -118.3658752, 77.6820831, -118.9976120, 78.3387527, -196.7046204, 196.6796875
18: -68.8117447, 41.8812180, -69.2970428, 41.9940491, -110.8057938, 111.1782455
19: -59.8389893, 24.8319893, -60.1339073, 24.8772297, -84.7162170, 84.9658966
20: -53.9304810, 32.1527176, -54.2105522, 32.2790909, -86.2095642, 86.3632660
21: -72.1449051, 36.4103470, -72.5061417, 36.5693054, -108.7142105, 108.9164886
22: -81.4472504, 47.6317711, -81.8364258, 47.9038353, -129.3510895, 129.4682007
23: -54.6257324, 34.5916443, -54.9616890, 34.6498108, -89.2755432, 89.5533295
24: -64.0468903, 34.5684814, -64.4236298, 34.6736755, -98.7205582, 98.9921036
25: -59.7938423, 39.4015503, -60.0729485, 39.5509148, -99.3447571, 99.4744873
26: -92.0780487, 49.8989944, -92.7406616, 50.4008827, -142.4789276, 142.6396484
27: -67.7935333, 44.1584435, -68.1596527, 44.2662506, -112.0597839, 112.3181000
28: -56.3336296, 36.4307938, -56.6094589, 36.5009308, -92.8345490, 93.0402527
29: -81.1510162, 53.7052155, -81.4665909, 54.0280571, -135.1790619, 135.1718140
30: -67.7329559, 36.6937141, -68.1449966, 36.8621864, -104.5951385, 104.8386993
31: -62.3369560, 30.5407314, -62.7350121, 30.5997925, -92.9367371, 93.2757416
32: -65.3153839, 47.6547623, -65.4703522, 47.8840179, -113.1993866, 113.1251144
33: -99.3313980, 58.2809486, -99.6749878, 58.5640411, -157.8954315, 157.9559326
34: -84.5981445, 44.3077583, -84.8934097, 44.5011406, -129.0992737, 129.2011566
35: -80.2310638, 47.2105446, -80.5044861, 47.4603577, -127.6914062, 127.7150269
36: -82.2282257, 48.3032074, -82.3892059, 48.4495811, -130.6778107, 130.6924133
37: -114.9179916, 47.9112701, -115.2475128, 48.0447426, -162.9627380, 163.1587830
38: -101.6438522, 63.3228683, -101.9564667, 63.5713654, -165.2152100, 165.2793274
39: -121.9596024, 54.5549431, -122.2373505, 54.8872681, -176.8468628, 176.7922974
40: -96.2360382, 47.2508469, -96.6040497, 47.4941559, -143.7301941, 143.8548889
41: -66.7547379, 39.6480217, -67.0017548, 39.8520546, -106.6067963, 106.6497726
42: -49.4857559, 44.3098450, -49.6231956, 44.5249176, -94.0106659, 93.9330444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1625
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1700
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
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1631
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
time: 378.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
time: 84.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.4443398, 46.0219879, -62.0942421, 45.8580017, -108.3023376, 108.1162262
1: -39.7334213, 41.8082542, -39.5317574, 41.7149887, -81.4483948, 81.3400116
2: -36.7968826, 43.8919754, -36.4774055, 43.7482681, -80.5451508, 80.3693848
3: -44.7590981, 52.0026169, -44.3782768, 51.8500671, -96.6091614, 96.3808823
4: -52.3151627, 40.5374107, -51.9666176, 40.4011993, -92.7163620, 92.5040283
5: -46.6152496, 57.0450363, -46.2591705, 56.8759766, -103.4912262, 103.3042068
6: -67.7209702, 41.4633484, -67.5661011, 41.3632584, -109.0842056, 109.0294495
7: -56.9593697, 52.9500656, -56.6570969, 52.8696976, -109.8290558, 109.6071625
8: -47.0418091, 47.1448441, -46.7721519, 46.9958076, -94.0375977, 93.9169922
9: -49.2889748, 52.4243164, -49.0920944, 52.2121162, -101.5010910, 101.5164108
10: -79.0425034, 76.0605316, -78.7061081, 75.3804398, -154.4229431, 154.7666321
11: -80.0546646, 52.4355583, -79.8831863, 51.9394455, -131.9941101, 132.3187408
12: -74.4825974, 58.1969452, -74.1812286, 57.6148453, -132.0974426, 132.3781738
13: -70.6501312, 66.2361374, -70.3994293, 66.1134796, -136.7636108, 136.6355591
14: -106.6434937, 56.7215958, -106.3003006, 56.2582130, -162.9017029, 163.0218964
15: -58.6593742, 50.4815369, -58.4235878, 50.3302841, -108.9896545, 108.9051208
16: -82.6838303, 66.0700226, -82.4685211, 65.7569275, -148.4407654, 148.5385437
17: -118.9173355, 77.9549408, -118.6441116, 77.3576508, -196.2749939, 196.5990448
18: -69.0374908, 41.9090462, -68.8445129, 41.5658913, -110.6033783, 110.7535553
19: -59.9808159, 24.7871151, -59.8285599, 24.5480003, -84.5288086, 84.6156769
20: -54.0884247, 32.1841240, -53.9230309, 31.9992237, -86.0876465, 86.1071548
21: -72.3197403, 36.4380760, -72.1103210, 36.0863800, -108.4061203, 108.5484009
22: -81.8105011, 47.8473282, -81.7212143, 47.5413628, -129.3518677, 129.5685425
23: -54.8038597, 34.5526962, -54.6770554, 34.3060608, -89.1099243, 89.2297516
24: -64.2487640, 34.6171188, -64.1712799, 34.5127220, -98.7614822, 98.7883987
25: -59.9407310, 39.5165596, -59.8774376, 39.3158112, -99.2565308, 99.3939896
26: -92.6951599, 50.3169174, -92.4341278, 49.7714157, -142.4665527, 142.7510376
27: -68.0328522, 44.2134933, -67.8740921, 44.0940323, -112.1268768, 112.0875854
28: -56.5011597, 36.4483719, -56.3835831, 36.3225746, -92.8237305, 92.8319550
29: -81.4513321, 53.9147491, -81.3448868, 53.5365372, -134.9878540, 135.2596436
30: -67.9247284, 36.7349968, -67.8005219, 36.4651718, -104.3899002, 104.5355148
31: -62.5189514, 30.5282955, -62.3314743, 30.3009720, -92.8199158, 92.8597717
32: -65.4249268, 47.7861938, -65.2733536, 47.6697540, -113.0946655, 113.0595398
33: -99.4710312, 58.3699036, -99.0981445, 58.1784668, -157.6495056, 157.4680481
34: -84.7965546, 44.4492416, -84.5167084, 44.3008423, -129.0973816, 128.9659424
35: -80.3497314, 47.3220596, -79.9780579, 47.1669350, -127.5166626, 127.3001175
36: -82.3134689, 48.3811646, -82.0175323, 48.2880707, -130.6015320, 130.3986969
37: -115.1225891, 48.0160599, -114.9310913, 47.8562393, -162.9788208, 162.9471436
38: -101.8882141, 63.4873886, -101.5654907, 63.3767166, -165.2649078, 165.0528870
39: -122.0399780, 54.7283630, -121.7169037, 54.6107941, -176.6507721, 176.4452515
40: -96.4728165, 47.4908524, -96.2180862, 47.4004669, -143.8732910, 143.7089386
41: -66.9456253, 39.7988510, -66.7938995, 39.6742439, -106.6198730, 106.5927505
42: -49.5875931, 44.3750343, -49.4691887, 44.0520439, -93.6396332, 93.8442230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1655
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
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1785
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
type: A, layer: 1, pos: 1431
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
type: A, layer: 1, pos: 1686
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
time: 67.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
time: 62.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.6455841, 46.0759201, -62.5897713, 46.1754379, -108.8210068, 108.6656876
1: -39.8471680, 41.8545609, -39.8027725, 41.8821640, -81.7293320, 81.6573334
2: -36.9883881, 43.9291840, -36.8935280, 43.9885712, -80.9769440, 80.8227081
3: -44.9787750, 52.0547256, -44.8529396, 52.2119789, -97.1907501, 96.9076614
4: -52.5421829, 40.5840034, -52.4647255, 40.6110535, -93.1532364, 93.0487289
5: -46.8139648, 57.0934563, -46.6965256, 57.2104950, -104.0244598, 103.7899780
6: -67.8028870, 41.5327606, -67.7942352, 41.5772057, -109.3800812, 109.3269958
7: -57.1084633, 52.9906921, -57.0140610, 53.0204697, -110.1289368, 110.0047531
8: -47.2055740, 47.1979904, -47.1385727, 47.2101097, -94.4156723, 94.3365631
9: -49.3931923, 52.5345764, -49.3701897, 52.4877892, -101.8809814, 101.9047699
10: -79.1425247, 76.4782486, -79.2648163, 76.2774582, -155.4199829, 155.7430725
11: -80.1362381, 52.8299942, -80.4054794, 52.7388496, -132.8750916, 133.2354736
12: -74.5411453, 58.5806503, -74.6886978, 58.4330444, -132.9741821, 133.2693481
13: -70.8262558, 66.3474121, -70.8220215, 66.5792618, -137.4055176, 137.1694336
14: -106.7704163, 57.0293655, -106.9233856, 56.9106636, -163.6810760, 163.9527588
15: -58.8267288, 50.5629654, -58.8538704, 50.5515709, -109.3782959, 109.4168396
16: -82.8097229, 66.2616425, -82.8468781, 66.1899414, -148.9996643, 149.1085205
17: -119.0119781, 78.4279175, -119.2555923, 78.3531189, -197.3650970, 197.6835022
18: -69.1421509, 42.1193008, -69.3840027, 42.0310860, -111.1732330, 111.5033035
19: -60.0509720, 24.9491615, -60.1939926, 24.8932266, -84.9441986, 85.1431580
20: -54.1569977, 32.3171539, -54.2761002, 32.2966156, -86.4536133, 86.5932541
21: -72.3958893, 36.6755753, -72.5819397, 36.5946579, -108.9905472, 109.2575073
22: -81.8907013, 48.0238609, -81.9906998, 47.9378586, -129.8285522, 130.0145416
23: -54.8654747, 34.7174683, -55.0178947, 34.6694870, -89.5349579, 89.7353516
24: -64.3248901, 34.6990585, -64.4652405, 34.7083282, -99.0332108, 99.1642914
25: -60.0055733, 39.6367645, -60.1289444, 39.5894127, -99.5949860, 99.7657089
26: -92.7736053, 50.6200409, -93.0124817, 50.4444809, -143.2180786, 143.6325226
27: -68.1513977, 44.3018112, -68.2011337, 44.3081512, -112.4595490, 112.5029373
28: -56.5659790, 36.5284538, -56.6574936, 36.5274506, -93.0934067, 93.1859436
29: -81.5144424, 54.1567459, -81.5924149, 54.0537491, -135.5681915, 135.7491608
30: -67.9950180, 36.9207306, -68.1924744, 36.8874359, -104.8824463, 105.1132050
31: -62.6257858, 30.6828766, -62.7923508, 30.6285992, -93.2543869, 93.4752197
32: -65.5156860, 47.8845978, -65.5180893, 47.9113541, -113.4270401, 113.4026871
33: -99.7521591, 58.4377708, -99.7203827, 58.6188393, -158.3710022, 158.1581421
34: -84.9774323, 44.5132141, -84.9363556, 44.5681190, -129.5455475, 129.4495697
35: -80.6054382, 47.3798599, -80.5398560, 47.5140076, -128.1194458, 127.9197083
36: -82.5040436, 48.4332428, -82.4381256, 48.4905777, -130.9946289, 130.8713684
37: -115.2603378, 48.0981522, -115.3077393, 48.0889282, -163.3492737, 163.4058838
38: -102.0667114, 63.5593491, -102.0041885, 63.6363564, -165.7030640, 165.5635376
39: -122.2807388, 54.7806931, -122.2818298, 54.9469986, -177.2277069, 177.0625305
40: -96.6411667, 47.5243301, -96.6436996, 47.6075363, -144.2487030, 144.1680298
41: -67.0359650, 39.8772926, -67.0373077, 39.9108505, -106.9468079, 106.9145889
42: -49.6542053, 44.5976257, -49.6675835, 44.5638580, -94.2180634, 94.2652054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1685
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
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
type: A, layer: 1, pos: 1342
type: A, layer: 1, pos: 871
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1686
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
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1393
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
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1429
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
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
time: 80.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8118448
time: 83.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.3498421, 45.8048553, -62.6191254, 45.9734192, -108.3232422, 108.4239731
1: -39.6495132, 41.6852264, -39.8419762, 41.8095779, -81.4590912, 81.5272064
2: -36.7455521, 43.6750641, -37.0364876, 43.8201370, -80.5656891, 80.7115479
3: -44.6725273, 51.7458153, -44.9516640, 51.9599075, -96.6324310, 96.6974792
4: -52.3727379, 40.4399605, -52.5915833, 40.5422478, -92.9149857, 93.0315399
5: -46.5011711, 56.7352295, -46.8260841, 56.9754448, -103.4766159, 103.5613098
6: -67.5872955, 41.3524437, -67.7536316, 41.6362495, -109.2235413, 109.1060791
7: -56.7583313, 52.6817932, -57.1164551, 52.9176445, -109.6759796, 109.7982483
8: -47.0049973, 46.9174728, -47.3325882, 47.0858574, -94.0908508, 94.2500534
9: -49.0426979, 52.2702942, -49.2077179, 52.6798172, -101.7225189, 101.4780121
10: -78.6051178, 75.9344788, -78.9345551, 76.4559479, -155.0610657, 154.8690338
11: -79.9722366, 52.4953995, -80.0663300, 52.7332268, -132.7054596, 132.5617371
12: -73.8388062, 57.9541397, -74.2465820, 58.7939758, -132.6327820, 132.2007141
13: -70.3983612, 65.9799805, -70.4835815, 66.4151459, -136.8135071, 136.4635620
14: -106.0139236, 56.3988876, -106.4774246, 56.9933815, -163.0072937, 162.8763123
15: -58.5869255, 50.3099022, -58.8674431, 50.5146751, -109.1016006, 109.1773453
16: -82.5540009, 66.0163498, -82.7077103, 66.2949524, -148.8489532, 148.7240601
17: -118.3725739, 77.7355652, -118.7453384, 78.3957214, -196.7682953, 196.4808960
18: -68.8272629, 41.8643761, -69.0774918, 41.9197121, -110.7469635, 110.9418640
19: -59.8511543, 24.8048210, -59.9949951, 24.7933941, -84.6445465, 84.7998047
20: -53.9474564, 32.1561127, -54.0928078, 32.2554398, -86.2028961, 86.2489166
21: -72.1562958, 36.3970795, -72.3180084, 36.5111847, -108.6674805, 108.7150879
22: -81.4825668, 47.6479378, -81.8348236, 47.9036789, -129.3862457, 129.4827576
23: -54.6402473, 34.5627823, -54.8172607, 34.5595093, -89.1997528, 89.3800430
24: -64.1091309, 34.5385017, -64.4255295, 34.5793915, -98.6885223, 98.9640350
25: -59.8035698, 39.3981247, -59.9958305, 39.5196533, -99.3232193, 99.3939514
26: -92.1002197, 49.9981766, -92.5435410, 50.5435638, -142.6437836, 142.5417175
27: -67.8682022, 44.1189880, -68.2388153, 44.1585159, -112.0267105, 112.3578033
28: -56.3466568, 36.4151459, -56.5223618, 36.4346390, -92.7812958, 92.9375076
29: -81.1780853, 53.7213745, -81.4312820, 54.0184402, -135.1965332, 135.1526489
30: -67.7422638, 36.7137756, -67.9500122, 36.8561478, -104.5984116, 104.6637878
31: -62.3650055, 30.4868279, -62.6089973, 30.4721012, -92.8371048, 93.0958252
32: -65.3244476, 47.7435684, -65.4402390, 48.0151901, -113.3396378, 113.1837997
33: -99.3413239, 58.3042679, -99.6446686, 58.3978996, -157.7392273, 157.9489441
34: -84.6071014, 44.3242035, -84.8641586, 44.4495544, -129.0566406, 129.1883545
35: -80.2433395, 47.2261200, -80.4755325, 47.3315697, -127.5748901, 127.7016525
36: -82.1888733, 48.3195457, -82.2816925, 48.4122391, -130.6011047, 130.6012268
37: -114.9385605, 47.9335861, -115.2085800, 48.0440178, -162.9825592, 163.1421661
38: -101.6831055, 63.3397598, -101.9647980, 63.5194473, -165.2025452, 165.3045654
39: -121.9345474, 54.5702591, -122.1218338, 54.7248726, -176.6594238, 176.6920929
40: -96.2768250, 47.2612305, -96.6200485, 47.4251289, -143.7019501, 143.8812866
41: -66.7775574, 39.6982498, -67.0053558, 39.8875275, -106.6650848, 106.7036057
42: -49.5021973, 44.4109955, -49.6098824, 44.6575661, -94.1597595, 94.0208740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1641
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
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1569

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8025933
time: 66.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8031394
time: 118.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -62.5541382, 45.8594971, -63.1341896, 46.2958069, -108.8499298, 108.9936829
1: -39.7653160, 41.7318306, -40.1279984, 41.9793472, -81.7446594, 81.8598251
2: -36.9401169, 43.7129059, -37.4712868, 44.0639038, -81.0040207, 81.1841888
3: -44.8948593, 51.7986526, -45.4448891, 52.3258591, -97.2207184, 97.2435455
4: -52.6036720, 40.4870071, -53.1157951, 40.7586784, -93.3623505, 93.6027985
5: -46.7019577, 56.7844276, -47.2788544, 57.3143196, -104.0162811, 104.0632782
6: -67.6694183, 41.4299774, -67.9950943, 41.8732986, -109.5427170, 109.4250641
7: -56.9088364, 52.7226334, -57.4922295, 53.0712662, -109.9801025, 110.2148438
8: -47.1719170, 46.9712982, -47.7194481, 47.3044052, -94.4763184, 94.6907501
9: -49.1485672, 52.3828239, -49.4951477, 52.9744759, -102.1230469, 101.8779678
10: -78.7068481, 76.3578796, -79.5001678, 77.3923721, -156.0992126, 155.8580475
11: -80.0531998, 52.8951149, -80.6237946, 53.5640945, -133.6172943, 133.5188904
12: -73.8983154, 58.3430367, -74.7604980, 59.6406555, -133.5389709, 133.1035309
13: -70.5820847, 66.0911407, -70.9303284, 66.8895645, -137.4716492, 137.0214539
14: -106.1427002, 56.7093811, -107.1091232, 57.6597824, -163.8024750, 163.8184814
15: -58.7703972, 50.3911819, -59.3548050, 50.7521744, -109.5225677, 109.7459869
16: -82.6799316, 66.2125092, -83.0973663, 66.7774734, -149.4573975, 149.3098755
17: -118.4688110, 78.2139664, -119.3651047, 79.4207458, -197.8895569, 197.5790558
18: -68.9327087, 42.0766220, -69.6244965, 42.3997421, -111.3324432, 111.7011032
19: -59.9220200, 24.9687004, -60.3658447, 25.1503639, -85.0723877, 85.3345413
20: -54.0170708, 32.2905731, -54.4550056, 32.5620422, -86.5791168, 86.7455750
21: -72.2335205, 36.6365738, -72.7965851, 37.0317078, -109.2652283, 109.4331512
22: -81.5626373, 47.8269119, -82.1135254, 48.3171883, -129.8798218, 129.9404297
23: -54.7018852, 34.7292328, -55.1657600, 34.9351425, -89.6370087, 89.8949890
24: -64.1858826, 34.6208992, -64.7195282, 34.7775993, -98.9634476, 99.3404236
25: -59.8688354, 39.5198174, -60.2524147, 39.8060989, -99.6749344, 99.7722321
26: -92.1801071, 50.3036728, -93.1293182, 51.2343979, -143.4144897, 143.4329834
27: -67.9880447, 44.2086182, -68.5764160, 44.3799438, -112.3679886, 112.7850342
28: -56.4114380, 36.4969139, -56.8011322, 36.6498680, -93.0613022, 93.2980499
29: -81.2429810, 53.9656677, -81.6893921, 54.5528870, -135.7958679, 135.6550598
30: -67.8126678, 36.9021683, -68.3541336, 37.2931366, -105.1058044, 105.2563019
31: -62.4716187, 30.6426697, -63.0761528, 30.8100986, -93.2817154, 93.7188263
32: -65.4157181, 47.8434830, -65.6975861, 48.2718086, -113.6875153, 113.5410690
33: -99.6250229, 58.3731461, -100.2795410, 58.8446732, -158.4696808, 158.6526794
34: -84.7892914, 44.3888474, -85.2940826, 44.7278748, -129.5171661, 129.6829224
35: -80.5015488, 47.2847366, -81.0539856, 47.6939125, -128.1954651, 128.3387146
36: -82.3809357, 48.3710060, -82.7209396, 48.6189842, -130.9999237, 131.0919495
37: -115.0779266, 48.0161667, -115.6013718, 48.2875710, -163.3654938, 163.6175385
38: -101.8663559, 63.4128532, -102.4302597, 63.7857513, -165.6520996, 165.8431091
39: -122.1779480, 54.6224442, -122.7076721, 55.0714493, -177.2493896, 177.3301086
40: -96.4472809, 47.2952309, -97.0605927, 47.6342812, -144.0815582, 144.3558197
41: -66.8686752, 39.7821465, -67.2605591, 40.1398544, -107.0085297, 107.0427094
42: -49.5682335, 44.6375046, -49.8173027, 45.2043304, -94.7725601, 94.4548035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=370, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1735
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
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1625
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
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1479
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
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
time: 96.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
time: 80.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 179.42 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7097297
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7662435
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8118448
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8025933
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.8031394
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 179.42
Output dim: 2, lower bound: -53.7587027, upper bound: 53.7595536
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8678289
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7244811
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8241711
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.8119656
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.9423880, upper bound: 53.8876334
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.7765001
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 179.42
Output dim: 2, lower bound: -53.8330470, upper bound: 53.9423876
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=81.74913024902344
rel_dist={2: [-53.9565580342401, 53.95655804785923]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

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
time: 68.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4987990, upper bound: 52.4987991
time: 69.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 138.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 138.42
Output dim: 2, lower bound: -52.4321920, upper bound: 52.4987991
IS_A2, status: Status.UNKNOWN, split count: 1, time: 138.42
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 69.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
time: 70.33 seconds

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

Time for backsubstitution: 2.30 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
time: 75.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
time: 78.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 156.00 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 156.00
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4030865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 156.00
Output dim: 2, lower bound: -52.4115784, upper bound: 52.4961799
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 156.00
Output dim: 2, lower bound: -52.4782228, upper bound: 52.4030865
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 156.00
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

Time for backsubstitution: 2.26 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4265702, upper bound: 52.4467989
time: 101.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
time: 76.22 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
time: 71.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753
time: 75.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 149.62 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 149.62
Output dim: 2, lower bound: -52.4265702, upper bound: 52.4467989
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 149.62
Output dim: 2, lower bound: -52.4093444, upper bound: 52.3995753
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 149.62
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3590837
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 149.62
Output dim: 2, lower bound: -52.4759374, upper bound: 52.3995753

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=370, inp2_unstable=371, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=444, inp2_unstable=444, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3810336
time: 81.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3540111
time: 79.28 seconds

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

Time for backsubstitution: 2.29 seconds

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
time: 64.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
time: 74.87 seconds

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

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
time: 76.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524130
time: 124.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 203.14 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 203.14
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3810336
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 203.14
Output dim: 2, lower bound: -52.3962307, upper bound: 52.3540111
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 203.14
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3081349
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 203.14
Output dim: 2, lower bound: -52.4733732, upper bound: 52.3540111
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 203.14
Output dim: 2, lower bound: -52.4631934, upper bound: 52.3524131
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 203.14
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

Time for backsubstitution: 2.38 seconds

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
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
time: 65.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 74.14 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
time: 78.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
time: 76.98 seconds

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

Time for backsubstitution: 2.34 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 76.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
time: 74.85 seconds

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
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
time: 75.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
time: 165.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 243.79 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966785
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3414505
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3415663
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3450704
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.3870460
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 243.79
Output dim: 2, lower bound: -52.3378656, upper bound: 52.2966831
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=81.74913024902344
rel_dist={2: [-52.50344204029277, 52.50344204220282]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 10158.58 seconds
