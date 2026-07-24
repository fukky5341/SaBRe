## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 55.7118901422


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877)
1: (-26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162)
2: (-26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880)
3: (-26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104)
4: (-36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473)
5: (-27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734)
6: (-56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708)
7: (-35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175)
8: (-47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871)
9: (-31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987)
10: (-45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319)
11: (-49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924)
12: (-31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817)
13: (-29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827)
14: (-67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113)
15: (-35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776)
16: (-55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120)
17: (-55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509)
18: (-60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796)
19: (-43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266)
20: (-40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051)
21: (-52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070)
22: (-41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907)
23: (-41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519)
24: (-55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559)
25: (-36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804)
26: (-56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965)
27: (-66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740)
28: (-41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535)
29: (-42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394)
30: (-51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581)
31: (-58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796)
32: (-45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751)
33: (-75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069)
34: (-62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241)
35: (-56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479)
36: (-54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594)
37: (-94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476)
38: (-71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703)
39: (-80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231)
40: (-79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684)
41: (-57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034)
42: (-35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 87.06 = 89.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -55.7676578, upper bound: 55.7676578

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964
time: 77.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964
time: 71.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 148.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 148.72
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964
IS_A2, status: Status.UNKNOWN, split count: 1, time: 148.72
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -62.0228348, 35.6752090, -62.0630913, 35.6837311, -97.7065659, 97.7382965
1: -26.3178959, 29.8744354, -26.3448658, 29.8817043, -56.1996002, 56.2192993
2: -26.3296261, 30.9630871, -26.3583164, 30.9695282, -57.2991562, 57.3214035
3: -26.1100845, 39.0832520, -26.1432972, 39.0914383, -65.2015228, 65.2265472
4: -36.4472046, 31.7181721, -36.4873199, 31.7252331, -68.1724396, 68.2054901
5: -27.7399025, 36.0128174, -27.7672253, 36.0223579, -63.7622604, 63.7800446
6: -56.4556618, 22.9075584, -56.4730148, 22.9238472, -79.3795090, 79.3805695
7: -35.5250778, 27.1062603, -35.5588646, 27.1154366, -62.6405144, 62.6651230
8: -47.2474022, 38.1418343, -47.2890358, 38.1529846, -85.4003906, 85.4308701
9: -31.4400520, 42.4755859, -31.4642372, 42.4835739, -73.9236298, 73.9398193
10: -45.5161400, 54.2853241, -45.5342789, 54.3001709, -99.8163147, 99.8196030
11: -49.1386452, 18.6984863, -49.1472511, 18.7289619, -67.8676071, 67.8457336
12: -31.1869297, 45.6078835, -31.2070141, 45.6268768, -76.8138046, 76.8148956
13: -29.6222191, 70.2063141, -29.6996727, 70.2167358, -99.8389587, 99.9059906
14: -67.4532623, 33.1723709, -67.4880524, 33.1787300, -100.6319885, 100.6604233
15: -35.4500656, 36.9726868, -35.4724960, 36.9871140, -72.4371796, 72.4451828
16: -54.9761963, 24.9645958, -54.9920959, 24.9761353, -79.9523315, 79.9566956
17: -55.4775314, 40.7146759, -55.5286217, 40.7238693, -96.2014008, 96.2433014
18: -60.4729881, 16.2441559, -60.4838562, 16.2968369, -76.7698212, 76.7280121
19: -43.0024452, 15.1625462, -43.0107956, 15.1970768, -58.1995239, 58.1733398
20: -40.3134537, 20.0654621, -40.3220825, 20.0938950, -60.4073486, 60.3875427
21: -51.9833069, 17.0530319, -51.9935036, 17.0929947, -69.0763016, 69.0465393
22: -41.6514359, 27.0544968, -41.6647377, 27.0854130, -68.7368469, 68.7192383
23: -41.4968948, 23.7449627, -41.5033989, 23.7869759, -65.2838745, 65.2483597
24: -55.3737106, 20.6874676, -55.3821144, 20.7279949, -76.1017075, 76.0695801
25: -36.4289627, 30.0776558, -36.4372635, 30.1196404, -66.5485992, 66.5149231
26: -56.7917976, 25.7886009, -56.8067780, 25.8385715, -82.6303711, 82.5953827
27: -66.7642670, 11.9984169, -66.7740021, 12.0490990, -78.8133698, 78.7724152
28: -41.3106499, 27.7953682, -41.3183060, 27.8360348, -69.1466827, 69.1136780
29: -42.6679878, 25.1889305, -42.6800499, 25.2209187, -67.8889084, 67.8689804
30: -51.1774521, 25.0180397, -51.1846199, 25.0503254, -76.2277756, 76.2026596
31: -58.0171738, 22.2642403, -58.0296593, 22.3110237, -80.3282013, 80.2938995
32: -45.0221100, 29.7262554, -45.0488625, 29.7364616, -74.7585754, 74.7751160
33: -75.8919907, 30.9171600, -75.9225693, 30.9290199, -106.8210144, 106.8397293
34: -62.0842781, 19.5713196, -62.1028976, 19.5786743, -81.6629486, 81.6742172
35: -56.8324585, 29.6078606, -56.8529587, 29.6180611, -86.4505157, 86.4608154
36: -54.9258118, 29.1202164, -54.9488258, 29.1282101, -84.0540237, 84.0690460
37: -94.3794556, 6.4937992, -94.4050598, 6.5039215, -100.8833771, 100.8988571
38: -71.5248032, 30.3228035, -71.5523682, 30.3319206, -101.8567200, 101.8751678
39: -80.0808716, 27.2942963, -80.1314240, 27.2996483, -107.3805237, 107.4257202
40: -79.7405090, 0.4925356, -79.7661285, 0.5008736, -80.2413788, 80.2586670
41: -57.2110748, 21.6400204, -57.2283325, 21.6501274, -78.8612061, 78.8683548
42: -35.8935890, 22.1430721, -35.9048920, 22.1566601, -58.0502472, 58.0479660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7529081
time: 94.15 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7548025
time: 78.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -62.1567764, 35.8052979, -62.1075897, 35.6926041, -97.8493805, 97.9128876
1: -26.4051266, 29.9736252, -26.3747635, 29.8893223, -56.2944489, 56.3483887
2: -26.4159660, 31.0677338, -26.3943939, 30.9762764, -57.3922424, 57.4621277
3: -26.2038517, 39.2190170, -26.1816998, 39.1002045, -65.3040543, 65.4007187
4: -36.5589142, 31.8329601, -36.5321007, 31.7324638, -68.2913818, 68.3650589
5: -27.8188057, 36.1508560, -27.7983742, 36.0328064, -63.8516121, 63.9492302
6: -56.5233459, 22.9931450, -56.4898682, 22.9449654, -79.4683075, 79.4830170
7: -35.6371956, 27.2294617, -35.5992889, 27.1253414, -62.7625351, 62.8287506
8: -47.3713608, 38.2794495, -47.3364563, 38.1653481, -85.5367126, 85.6159058
9: -31.5290833, 42.5636940, -31.4898930, 42.4915848, -74.0206680, 74.0535889
10: -45.6202583, 54.3513718, -45.5547676, 54.3153534, -99.9356079, 99.9061432
11: -49.3058853, 18.7890949, -49.1555634, 18.7643337, -68.0702209, 67.9446564
12: -31.2602940, 45.7349930, -31.2276630, 45.6481552, -76.9084473, 76.9626541
13: -29.8254642, 70.4805984, -29.7911282, 70.2275467, -100.0530090, 100.2717285
14: -67.6021347, 33.2562714, -67.5265808, 33.1855888, -100.7877197, 100.7828522
15: -35.5350227, 37.0216827, -35.4980545, 36.9999924, -72.5350189, 72.5197372
16: -55.0673294, 25.0396767, -55.0084305, 24.9879208, -80.0552521, 80.0481110
17: -55.6652222, 40.9165535, -55.5869675, 40.7340698, -96.3992920, 96.5035248
18: -60.6689682, 16.3863964, -60.4952011, 16.3577347, -77.0267029, 76.8815994
19: -43.1393051, 15.2506819, -43.0185089, 15.2371120, -58.3764191, 58.2691917
20: -40.4213181, 20.1461849, -40.3298264, 20.1266747, -60.5479927, 60.4760132
21: -52.1453896, 17.1612453, -52.0022659, 17.1395779, -69.2849655, 69.1635132
22: -41.7958031, 27.1393051, -41.6783600, 27.1212559, -68.9170609, 68.8176651
23: -41.6628036, 23.8608952, -41.5090332, 23.8360481, -65.4988556, 65.3699265
24: -55.5249329, 20.7933445, -55.3902359, 20.7749100, -76.2998428, 76.1835785
25: -36.5650635, 30.1943359, -36.4450874, 30.1686382, -66.7337036, 66.6394196
26: -56.9848938, 25.9169426, -56.8217888, 25.8957272, -82.8806229, 82.7387314
27: -66.9596176, 12.1245842, -66.7830353, 12.1079607, -79.0675812, 78.9076233
28: -41.4522285, 27.9024696, -41.3251038, 27.8831768, -69.3354034, 69.2275696
29: -42.8214760, 25.2779236, -42.6919899, 25.2594090, -68.0808868, 67.9699097
30: -51.2869148, 25.1158733, -51.1909256, 25.0878639, -76.3747787, 76.3068008
31: -58.2106018, 22.3870926, -58.0420151, 22.3652039, -80.5758057, 80.4291077
32: -45.1184196, 29.8081856, -45.0789032, 29.7487526, -74.8671722, 74.8870850
33: -75.9929962, 31.0649853, -75.9543610, 30.9417439, -106.9347382, 107.0193481
34: -62.1559753, 19.6316757, -62.1226044, 19.5856171, -81.7415924, 81.7542801
35: -56.9005814, 29.7032413, -56.8728790, 29.6295319, -86.5301132, 86.5761185
36: -55.0029373, 29.1809483, -54.9724426, 29.1367302, -84.1396637, 84.1533890
37: -94.4905243, 6.5690536, -94.4331665, 6.5148182, -101.0053406, 101.0022202
38: -71.6518478, 30.4003944, -71.5834961, 30.3417377, -101.9935837, 101.9838867
39: -80.2453003, 27.4652405, -80.1921692, 27.3046818, -107.5499802, 107.6574097
40: -79.8534546, 0.5892115, -79.7942810, 0.5098038, -80.3632584, 80.3834915
41: -57.2850533, 21.6790123, -57.2474518, 21.6620083, -78.9470596, 78.9264679
42: -35.9443207, 22.1962814, -35.9171028, 22.1734467, -58.1177673, 58.1133842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=368, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 693

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7529081
time: 69.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7548025
time: 79.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 150.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 150.97
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7529081
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 150.97
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7548025
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 150.97
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7529081
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 150.97
Output dim: 13, lower bound: -55.6344404, upper bound: 55.7548025

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -61.9692421, 35.6242256, -61.9754295, 35.5992546, -97.5684967, 97.5996552
1: -26.2743587, 29.8457279, -26.2733345, 29.8352509, -56.1096115, 56.1190643
2: -26.3027897, 30.9385262, -26.3148880, 30.9292545, -57.2320442, 57.2534142
3: -26.0686569, 39.0542221, -26.0765858, 39.0443649, -65.1130219, 65.1308060
4: -36.4215393, 31.6921921, -36.4455109, 31.6828041, -68.1043396, 68.1377029
5: -27.6709042, 35.9886284, -27.6556511, 35.9827576, -63.6536636, 63.6442795
6: -56.4351730, 22.8798866, -56.4397087, 22.8792019, -79.3143768, 79.3195953
7: -35.4351044, 27.0873013, -35.4121475, 27.0844650, -62.5195694, 62.4994507
8: -47.2040939, 38.1159019, -47.2190475, 38.1104431, -85.3145370, 85.3349457
9: -31.3882542, 42.4531136, -31.3798294, 42.4474716, -73.8357239, 73.8329468
10: -45.3916931, 54.2556686, -45.3308182, 54.2522392, -99.6439362, 99.5864868
11: -49.0019722, 18.6749229, -48.9236259, 18.6905785, -67.6925507, 67.5985489
12: -31.1536922, 45.5047073, -31.1525421, 45.4566345, -76.6103287, 76.6572495
13: -29.5785294, 70.0894852, -29.6286278, 70.0254364, -99.6039658, 99.7181091
14: -67.3908615, 33.1502876, -67.3871155, 33.1425552, -100.5334167, 100.5373993
15: -35.4184952, 36.9543381, -35.4210205, 36.9572601, -72.3757553, 72.3753586
16: -54.8722191, 24.9391823, -54.8209572, 24.9351292, -79.8073502, 79.7601395
17: -55.3785019, 40.6138878, -55.3670006, 40.5584602, -95.9369659, 95.9808884
18: -60.4008865, 16.2102985, -60.3651085, 16.2426987, -76.6435852, 76.5754089
19: -42.9324417, 15.1442146, -42.8957176, 15.1673927, -58.0998344, 58.0399323
20: -40.2361221, 20.0432396, -40.1936798, 20.0577412, -60.2938614, 60.2369194
21: -51.8723145, 17.0326271, -51.8134422, 17.0599861, -68.9322968, 68.8460693
22: -41.5814514, 27.0314274, -41.5492554, 27.0479603, -68.6294098, 68.5806808
23: -41.4425659, 23.7230930, -41.4147339, 23.7513695, -65.1939392, 65.1378250
24: -55.2920341, 20.6593227, -55.2476082, 20.6821575, -75.9741898, 75.9069290
25: -36.3660011, 30.0558929, -36.3335114, 30.0842590, -66.4502563, 66.3894043
26: -56.7385292, 25.7571964, -56.7189941, 25.7878017, -82.5263290, 82.4761887
27: -66.6611023, 11.9711323, -66.6041718, 12.0047855, -78.6658859, 78.5753021
28: -41.2781601, 27.7718487, -41.2650070, 27.7977123, -69.0758743, 69.0368576
29: -42.5565071, 25.1708450, -42.4960785, 25.1913719, -67.7478790, 67.6669235
30: -51.0662689, 24.9935951, -51.0015869, 25.0106125, -76.0768814, 75.9951782
31: -57.9305267, 22.2406673, -57.8889160, 22.2729301, -80.2034607, 80.1295853
32: -44.9940948, 29.6585579, -45.0030861, 29.6251755, -74.6192703, 74.6616440
33: -75.8650589, 30.8041458, -75.8785324, 30.7458496, -106.6109085, 106.6826782
34: -62.0666924, 19.4477940, -62.0745201, 19.3770714, -81.4437637, 81.5223160
35: -56.8091774, 29.4611092, -56.8150482, 29.3805122, -86.1896896, 86.2761536
36: -54.9034386, 29.0007324, -54.9122391, 28.9343166, -83.8377533, 83.9129715
37: -94.3353958, 6.3768597, -94.3331757, 6.3123398, -100.6477356, 100.7100372
38: -71.4986038, 30.1646996, -71.5099030, 30.0734863, -101.5720901, 101.6746063
39: -80.0459213, 27.1617508, -80.0742798, 27.0809174, -107.1268387, 107.2360306
40: -79.7009583, 0.4212885, -79.7014236, 0.3844538, -80.0854111, 80.1227112
41: -57.1918716, 21.5726376, -57.1969757, 21.5390930, -78.7309647, 78.7696152
42: -35.8678741, 22.1206074, -35.8629913, 22.1201935, -57.9880676, 57.9835968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=475, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
time: 67.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
time: 72.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -62.0143509, 35.6688995, -62.0914421, 35.6955109, -97.7098618, 97.7603455
1: -26.3130817, 29.8673935, -26.3482971, 29.9000168, -56.2130966, 56.2156906
2: -26.3233757, 30.9568691, -26.3658104, 30.9758186, -57.2991943, 57.3226776
3: -26.1026592, 39.0693779, -26.1431808, 39.1087151, -65.2113724, 65.2125549
4: -36.4426346, 31.7104740, -36.5035706, 31.7384014, -68.1810379, 68.2140427
5: -27.7335663, 36.0090866, -27.7699852, 36.0599289, -63.7934952, 63.7790718
6: -56.4488754, 22.9031906, -56.4995689, 22.9336281, -79.3825073, 79.4027557
7: -35.5184021, 27.1023197, -35.5661583, 27.1569786, -62.6753807, 62.6684799
8: -47.2402267, 38.1378326, -47.2940826, 38.1990128, -85.4392395, 85.4319153
9: -31.4285698, 42.4724655, -31.4739113, 42.5016212, -73.9301910, 73.9463806
10: -45.4955254, 54.2797203, -45.5350685, 54.3181000, -99.8136292, 99.8147888
11: -49.1248550, 18.6923065, -49.1528816, 18.7821941, -67.9070511, 67.8451843
12: -31.1816921, 45.6006126, -31.3451462, 45.6256485, -76.8073425, 76.9457550
13: -29.6139011, 70.1979675, -29.8434620, 70.2215118, -99.8354111, 100.0414276
14: -67.4443359, 33.1639061, -67.5126801, 33.1785316, -100.6228638, 100.6765900
15: -35.4447060, 36.9683609, -35.4839516, 37.0104752, -72.4551849, 72.4523163
16: -54.9580421, 24.9603348, -54.9997101, 25.0082741, -79.9663162, 79.9600449
17: -55.4674149, 40.7047501, -55.5564575, 40.7348442, -96.2022552, 96.2612076
18: -60.4648056, 16.2352562, -60.4903793, 16.3274632, -76.7922668, 76.7256317
19: -42.9943466, 15.1576643, -43.0207443, 15.2169294, -58.2112770, 58.1784096
20: -40.3077812, 20.0599270, -40.3346252, 20.1820621, -60.4898453, 60.3945541
21: -51.9738159, 17.0469723, -52.0125275, 17.1814442, -69.1552582, 69.0595016
22: -41.6445160, 27.0489941, -41.7030525, 27.0994492, -68.7439651, 68.7520447
23: -41.4896126, 23.7404690, -41.5084610, 23.8282928, -65.3179016, 65.2489319
24: -55.3666000, 20.6791420, -55.4006882, 20.7805252, -76.1471252, 76.0798340
25: -36.4237099, 30.0724926, -36.4554100, 30.1749420, -66.5986481, 66.5279007
26: -56.7855759, 25.7825470, -56.8338852, 25.8908806, -82.6764526, 82.6164322
27: -66.7557373, 11.9924297, -66.7856293, 12.1367445, -78.8924789, 78.7780609
28: -41.3059807, 27.7916298, -41.3193665, 27.8587761, -69.1647568, 69.1109924
29: -42.6581192, 25.1857719, -42.7131691, 25.2331429, -67.8912659, 67.8989410
30: -51.1697464, 25.0116844, -51.2013893, 25.1761017, -76.3458481, 76.2130737
31: -58.0065231, 22.2571793, -58.0539665, 22.3467445, -80.3532715, 80.3111420
32: -45.0158577, 29.7212143, -45.1104393, 29.7367878, -74.7526474, 74.8316498
33: -75.8876648, 30.9072762, -76.0229950, 30.9244328, -106.8120956, 106.9302673
34: -62.0786018, 19.5572910, -62.1067276, 19.5884876, -81.6670914, 81.6640167
35: -56.8267822, 29.5988712, -56.9663162, 29.6193962, -86.4461823, 86.5651855
36: -54.9189529, 29.1123524, -55.0599098, 29.1321487, -84.0511017, 84.1722641
37: -94.3726501, 6.4850540, -94.5342331, 6.4967918, -100.8694458, 101.0192871
38: -71.5173645, 30.3124275, -71.6404343, 30.3433571, -101.8607178, 101.9528656
39: -80.0752258, 27.2848911, -80.2534790, 27.2923317, -107.3675537, 107.5383682
40: -79.7237473, 0.4872122, -79.8064575, 0.5038271, -80.2275772, 80.2936707
41: -57.2053146, 21.6350765, -57.3069229, 21.6545963, -78.8599091, 78.9420013
42: -35.8829002, 22.1401234, -35.9255142, 22.1624985, -58.0453987, 58.0656357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242
time: 68.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242
time: 129.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -62.1032219, 35.7543373, -62.0199471, 35.6081085, -97.7113342, 97.7742844
1: -26.3616142, 29.9449196, -26.3032436, 29.8428726, -56.2044868, 56.2481613
2: -26.3891144, 31.0431747, -26.3509655, 30.9360085, -57.3251228, 57.3941422
3: -26.1624584, 39.1898880, -26.1150532, 39.0531235, -65.2155838, 65.3049393
4: -36.5332489, 31.8069801, -36.4902802, 31.6900215, -68.2232666, 68.2972565
5: -27.7498169, 36.1266327, -27.6868725, 35.9931641, -63.7429810, 63.8135071
6: -56.5028572, 22.9654922, -56.4565697, 22.9003448, -79.4031982, 79.4220581
7: -35.5472488, 27.2104607, -35.4525833, 27.0943699, -62.6416168, 62.6630440
8: -47.3280106, 38.2534790, -47.2664490, 38.1227722, -85.4507828, 85.5199280
9: -31.4773197, 42.5410843, -31.4055214, 42.4554749, -73.9327927, 73.9466095
10: -45.4957962, 54.3217163, -45.3513374, 54.2673798, -99.7631760, 99.6730499
11: -49.1691017, 18.7655888, -48.9319153, 18.7259254, -67.8950272, 67.6975021
12: -31.2271500, 45.6317825, -31.1731377, 45.4779701, -76.7051239, 76.8049164
13: -29.7818623, 70.3637390, -29.7201233, 70.0362244, -99.8180847, 100.0838623
14: -67.5398560, 33.2340965, -67.4256592, 33.1494408, -100.6893005, 100.6597595
15: -35.5034027, 37.0032883, -35.4465446, 36.9701118, -72.4735107, 72.4498291
16: -54.9633484, 25.0142059, -54.8372726, 24.9469299, -79.9102783, 79.8514786
17: -55.5661430, 40.8157654, -55.4254379, 40.5687103, -96.1348572, 96.2412033
18: -60.5968246, 16.3525963, -60.3764191, 16.3035641, -76.9003906, 76.7290192
19: -43.0693054, 15.2323551, -42.9034271, 15.2073956, -58.2767029, 58.1357803
20: -40.3439713, 20.1239967, -40.2014389, 20.0905304, -60.4345016, 60.3254356
21: -52.0343475, 17.1408787, -51.8221512, 17.1065750, -69.1409225, 68.9630280
22: -41.7257843, 27.1162357, -41.5628700, 27.0838223, -68.8096085, 68.6791077
23: -41.6084480, 23.8390350, -41.4203796, 23.8004341, -65.4088821, 65.2594147
24: -55.4432678, 20.7652245, -55.2557297, 20.7290859, -76.1723557, 76.0209503
25: -36.5020905, 30.1725655, -36.3413315, 30.1332531, -66.6353455, 66.5139008
26: -56.9315948, 25.8855438, -56.7340393, 25.8449631, -82.7765579, 82.6195831
27: -66.8564301, 12.0972757, -66.6132050, 12.0636539, -78.9200821, 78.7104797
28: -41.4197693, 27.8789673, -41.2717743, 27.8448200, -69.2645874, 69.1507416
29: -42.7099266, 25.2598419, -42.5080528, 25.2298698, -67.9397964, 67.7678986
30: -51.1757202, 25.0914383, -51.0078659, 25.0481434, -76.2238617, 76.0993042
31: -58.1239319, 22.3635502, -57.9012718, 22.3271332, -80.4510651, 80.2648239
32: -45.0904312, 29.7405148, -45.0331001, 29.6374512, -74.7278824, 74.7736130
33: -75.9660950, 30.9518661, -75.9103241, 30.7585163, -106.7246094, 106.8621902
34: -62.1383972, 19.5081921, -62.0941849, 19.3840828, -81.5224762, 81.6023788
35: -56.8773270, 29.5564156, -56.8349419, 29.3919373, -86.2692642, 86.3913574
36: -54.9805260, 29.0614643, -54.9358711, 28.9428444, -83.9233704, 83.9973373
37: -94.4463882, 6.4521856, -94.3612671, 6.3232603, -100.7696457, 100.8134537
38: -71.6256409, 30.2423325, -71.5409698, 30.0832939, -101.7089386, 101.7833023
39: -80.2103729, 27.3326721, -80.1350250, 27.0859871, -107.2963562, 107.4676971
40: -79.8138885, 0.5179157, -79.7295761, 0.3933935, -80.2072830, 80.2474899
41: -57.2658234, 21.6116619, -57.2160912, 21.5509968, -78.8168182, 78.8277512
42: -35.9186096, 22.1738281, -35.8751907, 22.1369705, -58.0555801, 58.0490189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
time: 396.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
time: 72.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -62.1482773, 35.7990112, -62.1359253, 35.7043381, -97.8526154, 97.9349365
1: -26.4003239, 29.9665966, -26.3782139, 29.9076271, -56.3079529, 56.3448105
2: -26.4096775, 31.0615158, -26.4019222, 30.9825821, -57.3922577, 57.4634399
3: -26.1964512, 39.2051010, -26.1816254, 39.1175003, -65.3139496, 65.3867264
4: -36.5543442, 31.8252926, -36.5483475, 31.7456245, -68.2999725, 68.3736420
5: -27.8124580, 36.1471176, -27.8011818, 36.0704117, -63.8828697, 63.9482994
6: -56.5165901, 22.9888210, -56.5163956, 22.9547653, -79.4713593, 79.5052185
7: -35.6305428, 27.2255096, -35.6065903, 27.1668758, -62.7974167, 62.8320999
8: -47.3641586, 38.2754211, -47.3415222, 38.2113800, -85.5755386, 85.6169434
9: -31.5176430, 42.5605621, -31.4995918, 42.5096397, -74.0272827, 74.0601501
10: -45.5995941, 54.3457985, -45.5555611, 54.3332939, -99.9328918, 99.9013596
11: -49.2921295, 18.7829342, -49.1611633, 18.8175716, -68.1097031, 67.9440994
12: -31.2551193, 45.7276497, -31.3657780, 45.6469574, -76.9020767, 77.0934296
13: -29.8171806, 70.4721985, -29.9349327, 70.2323685, -100.0495453, 100.4071350
14: -67.5932312, 33.2477722, -67.5511780, 33.1854973, -100.7787323, 100.7989502
15: -35.5296440, 37.0173416, -35.5095100, 37.0233231, -72.5529633, 72.5268555
16: -55.0491791, 25.0353909, -55.0159950, 25.0200729, -80.0692520, 80.0513840
17: -55.6550980, 40.9066429, -55.6148834, 40.7450638, -96.4001617, 96.5215302
18: -60.6607971, 16.3775082, -60.5017395, 16.3883591, -77.0491562, 76.8792496
19: -43.1312218, 15.2457905, -43.0284348, 15.2569275, -58.3881493, 58.2742233
20: -40.4156456, 20.1406364, -40.3423843, 20.2148724, -60.6305161, 60.4830208
21: -52.1358948, 17.1552410, -52.0212936, 17.2280121, -69.3639069, 69.1765366
22: -41.7888680, 27.1337776, -41.7166176, 27.1353188, -68.9241867, 68.8503952
23: -41.6555061, 23.8563576, -41.5141068, 23.8773804, -65.5328827, 65.3704681
24: -55.5178185, 20.7850285, -55.4088402, 20.8274879, -76.3453064, 76.1938705
25: -36.5598297, 30.1891479, -36.4632149, 30.2239704, -66.7837982, 66.6523590
26: -56.9786263, 25.9108658, -56.8488922, 25.9480057, -82.9266357, 82.7597580
27: -66.9510727, 12.1185493, -66.7946854, 12.1955891, -79.1466599, 78.9132385
28: -41.4475784, 27.8987198, -41.3261452, 27.9059486, -69.3535309, 69.2248688
29: -42.8115997, 25.2747707, -42.7250938, 25.2716293, -68.0832291, 67.9998627
30: -51.2792091, 25.1095390, -51.2077332, 25.2136574, -76.4928665, 76.3172760
31: -58.1999702, 22.3800907, -58.0662766, 22.4009476, -80.6009216, 80.4463654
32: -45.1121674, 29.8031616, -45.1404915, 29.7490597, -74.8612289, 74.9436493
33: -75.9886780, 31.0550423, -76.0548248, 30.9372101, -106.9258881, 107.1098633
34: -62.1503029, 19.6176586, -62.1264496, 19.5954475, -81.7457504, 81.7441101
35: -56.8949203, 29.6942654, -56.9863129, 29.6308422, -86.5257645, 86.6805801
36: -54.9960861, 29.1730728, -55.0835915, 29.1406746, -84.1367645, 84.2566681
37: -94.4836731, 6.5603247, -94.5623703, 6.5076895, -100.9913635, 101.1226959
38: -71.6444321, 30.3900299, -71.6715622, 30.3531609, -101.9975891, 102.0615921
39: -80.2396088, 27.4558792, -80.3142624, 27.2973652, -107.5369720, 107.7701416
40: -79.8366547, 0.5838833, -79.8346405, 0.5127783, -80.3494339, 80.4185257
41: -57.2792664, 21.6741104, -57.3260498, 21.6664696, -78.9457397, 79.0001602
42: -35.9336243, 22.1933403, -35.9377136, 22.1792564, -58.1128807, 58.1310539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=367, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6988193, upper bound: 55.7512242
time: 70.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242
time: 130.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 203.62 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7493188
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.6988193, upper bound: 55.7512242
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 203.62
Output dim: 13, lower bound: -55.5784404, upper bound: 55.7512242

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -61.9318161, 35.6105804, -61.9541473, 35.5915794, -97.5233917, 97.5647278
1: -26.2355289, 29.8362675, -26.2515945, 29.8299103, -56.0654373, 56.0878601
2: -26.2678185, 30.9264431, -26.2952709, 30.9224586, -57.1902771, 57.2217140
3: -26.0176849, 39.0433044, -26.0481300, 39.0380936, -65.0557785, 65.0914307
4: -36.3981400, 31.6815205, -36.4321976, 31.6765480, -68.0746918, 68.1137161
5: -27.6305809, 35.9774246, -27.6329346, 35.9763718, -63.6069527, 63.6103592
6: -56.3962097, 22.8657475, -56.4175949, 22.8710213, -79.2672272, 79.2833405
7: -35.3820801, 27.0764389, -35.3824463, 27.0783157, -62.4603958, 62.4588852
8: -47.1844254, 38.0949860, -47.2078400, 38.0985565, -85.2829819, 85.3028259
9: -31.3437481, 42.4443283, -31.3546543, 42.4420853, -73.7858353, 73.7989807
10: -45.3206444, 54.2371216, -45.2911797, 54.2416458, -99.5622864, 99.5283051
11: -48.9718056, 18.6582146, -48.9057198, 18.6811523, -67.6529541, 67.5639343
12: -31.1092033, 45.4820938, -31.1275826, 45.4437294, -76.5529327, 76.6096802
13: -29.4470177, 70.0716248, -29.5556297, 70.0152206, -99.4622345, 99.6272583
14: -67.3591614, 33.1106949, -67.3691483, 33.1203651, -100.4795227, 100.4798431
15: -35.3952446, 36.9077072, -35.4076385, 36.9299088, -72.3251495, 72.3153458
16: -54.8238754, 24.9190407, -54.7934227, 24.9238529, -79.7477264, 79.7124634
17: -55.3449249, 40.5781403, -55.3476639, 40.5382843, -95.8832092, 95.9258041
18: -60.3823700, 16.0747814, -60.3546371, 16.1665230, -76.5488892, 76.4294205
19: -42.9150162, 15.1055899, -42.8857880, 15.1459026, -58.0609207, 57.9913788
20: -40.2197609, 19.9917603, -40.1844101, 20.0288200, -60.2485809, 60.1761703
21: -51.8476715, 16.9900627, -51.7994995, 17.0361271, -68.8837967, 68.7895660
22: -41.5582390, 26.9514542, -41.5361595, 27.0029945, -68.5612335, 68.4876099
23: -41.4306145, 23.6758671, -41.4079056, 23.7249470, -65.1555634, 65.0837708
24: -55.2791977, 20.5985031, -55.2401772, 20.6479244, -75.9271240, 75.8386841
25: -36.3518677, 30.0023251, -36.3255005, 30.0541801, -66.4060516, 66.3278275
26: -56.7197876, 25.6258106, -56.7083626, 25.7149200, -82.4347076, 82.3341751
27: -66.6440582, 11.8705330, -66.5944672, 11.9489717, -78.5930328, 78.4649963
28: -41.2647705, 27.6982231, -41.2573700, 27.7566853, -69.0214539, 68.9555969
29: -42.5356140, 25.1173019, -42.4842949, 25.1614799, -67.6970978, 67.6015930
30: -51.0485420, 24.9432087, -50.9914474, 24.9821892, -76.0307312, 75.9346542
31: -57.9057541, 22.1958809, -57.8747864, 22.2474384, -80.1531906, 80.0706635
32: -44.9353523, 29.6500702, -44.9703026, 29.6202755, -74.5556259, 74.6203766
33: -75.8131332, 30.7809429, -75.8494415, 30.7325630, -106.5457001, 106.6303864
34: -62.0401077, 19.4285774, -62.0595779, 19.3662777, -81.4063873, 81.4881592
35: -56.7666397, 29.4442635, -56.7909431, 29.3710976, -86.1377411, 86.2352066
36: -54.8659859, 28.9896545, -54.8910141, 28.9280624, -83.7940521, 83.8806686
37: -94.3134155, 6.3570271, -94.3203888, 6.3012581, -100.6146698, 100.6774139
38: -71.4539795, 30.1482677, -71.4848251, 30.0642204, -101.5182037, 101.6330948
39: -79.9494629, 27.1421738, -80.0205536, 27.0698471, -107.0193100, 107.1627274
40: -79.6762238, 0.4006596, -79.6873474, 0.3729134, -80.0491333, 80.0880051
41: -57.1421127, 21.5594921, -57.1688232, 21.5316772, -78.6737900, 78.7283173
42: -35.8205719, 22.1049709, -35.8362045, 22.1113091, -57.9318810, 57.9411774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=474, inp2_unstable=475, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
time: 75.26 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
time: 70.50 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -62.0007057, 35.6601295, -61.9668846, 35.5971756, -97.5978851, 97.6270142
1: -26.2826500, 29.8967514, -26.2687302, 29.8338661, -56.1165161, 56.1654816
2: -26.3117981, 30.9881210, -26.3102741, 30.9277210, -57.2395172, 57.2983932
3: -26.0892296, 39.1134109, -26.0696239, 39.0417671, -65.1309967, 65.1830368
4: -36.4410934, 31.7274857, -36.4428177, 31.6809692, -68.1220627, 68.1703033
5: -27.6847839, 36.0454178, -27.6495056, 35.9808731, -63.6656570, 63.6949234
6: -56.4644279, 22.9712868, -56.4351654, 22.8771839, -79.3416138, 79.4064484
7: -35.4466743, 27.1471939, -35.4053497, 27.0828285, -62.5295029, 62.5525436
8: -47.2362366, 38.1501160, -47.2170219, 38.1070328, -85.3432693, 85.3671417
9: -31.4114647, 42.5226440, -31.3757076, 42.4453506, -73.8568115, 73.8983536
10: -45.4186630, 54.3815842, -45.3239136, 54.2495079, -99.6681671, 99.7054977
11: -49.0426903, 18.6959419, -48.9186172, 18.6879005, -67.7305908, 67.6145630
12: -31.1753445, 45.6006241, -31.1478062, 45.4545822, -76.6299286, 76.7484283
13: -29.5957508, 70.2803268, -29.6174278, 70.0224075, -99.6181564, 99.8977509
14: -67.4584503, 33.1642113, -67.3826828, 33.1331482, -100.5915985, 100.5468903
15: -35.4876862, 36.9742966, -35.4180145, 36.9523087, -72.4399948, 72.3923111
16: -54.8975487, 25.0519180, -54.8075943, 24.9324284, -79.8299789, 79.8595123
17: -55.4497948, 40.6423416, -55.3608627, 40.5511398, -96.0009308, 96.0032043
18: -60.5487137, 16.2291107, -60.3614502, 16.2305508, -76.7792664, 76.5905609
19: -43.0039749, 15.1562691, -42.8930168, 15.1638098, -58.1677856, 58.0492859
20: -40.3109360, 20.0505733, -40.1916275, 20.0528774, -60.3638153, 60.2422028
21: -51.9457817, 17.0422783, -51.8098145, 17.0558052, -69.0015869, 68.8520966
22: -41.7430000, 27.0432568, -41.5457726, 27.0412140, -68.7842102, 68.5890274
23: -41.5292511, 23.7319508, -41.4124947, 23.7468224, -65.2760773, 65.1444473
24: -55.4181366, 20.6735249, -55.2446098, 20.6756706, -76.0938110, 75.9181366
25: -36.4500046, 30.0656948, -36.3314819, 30.0785179, -66.5285187, 66.3971786
26: -56.9227486, 25.7722187, -56.7159729, 25.7767277, -82.6994781, 82.4881897
27: -66.8332977, 11.9773769, -66.6013412, 11.9958105, -78.8291092, 78.5787201
28: -41.3928909, 27.7781181, -41.2632217, 27.7910824, -69.1839752, 69.0413361
29: -42.6980324, 25.1783371, -42.4920578, 25.1863384, -67.8843689, 67.6703949
30: -51.1613083, 25.0037384, -50.9989128, 25.0047398, -76.1660461, 76.0026550
31: -58.0123100, 22.2564125, -57.8849831, 22.2674332, -80.2797394, 80.1413956
32: -45.0312729, 29.7349358, -44.9982262, 29.6237831, -74.6550598, 74.7331619
33: -75.8887100, 30.8797531, -75.8721390, 30.7425003, -106.6312103, 106.7518921
34: -62.0963211, 19.4805317, -62.0715027, 19.3742542, -81.4705734, 81.5520325
35: -56.8298378, 29.5270138, -56.8101883, 29.3774281, -86.2072678, 86.3372040
36: -54.9306450, 29.0380993, -54.9082413, 28.9327145, -83.8633575, 83.9463425
37: -94.3756409, 6.4173679, -94.3292923, 6.3098488, -100.6854858, 100.7466583
38: -71.5224915, 30.2393417, -71.5042267, 30.0708561, -101.5933456, 101.7435684
39: -80.0761108, 27.3010139, -80.0659637, 27.0780659, -107.1541748, 107.3669739
40: -79.7597427, 0.4733849, -79.6987610, 0.3810062, -80.1407471, 80.1721497
41: -57.2223663, 21.6290665, -57.1917686, 21.5375977, -78.7599640, 78.8208313
42: -35.8993721, 22.1981010, -35.8583145, 22.1178627, -58.0172348, 58.0564156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=475, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
time: 79.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
time: 73.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -61.9769020, 35.6552315, -62.0701485, 35.6878090, -97.6647110, 97.7253799
1: -26.2742825, 29.8579254, -26.3265495, 29.8946686, -56.1689529, 56.1844749
2: -26.2883644, 30.9447498, -26.3462334, 30.9690514, -57.2574158, 57.2909851
3: -26.0516891, 39.0585022, -26.1147232, 39.1024513, -65.1541443, 65.1732254
4: -36.4192581, 31.6997967, -36.4902687, 31.7321835, -68.1514435, 68.1900635
5: -27.6932755, 35.9979019, -27.7473106, 36.0535126, -63.7467880, 63.7452126
6: -56.4098930, 22.8891220, -56.4774513, 22.9254456, -79.3353424, 79.3665771
7: -35.4653740, 27.0914345, -35.5364113, 27.1508331, -62.6162071, 62.6278458
8: -47.2206039, 38.1168671, -47.2829247, 38.1871262, -85.4077301, 85.3997955
9: -31.3841019, 42.4637070, -31.4487400, 42.4962234, -73.8803253, 73.9124451
10: -45.4244499, 54.2611656, -45.4954300, 54.3075333, -99.7319794, 99.7565918
11: -49.0946312, 18.6755810, -49.1349449, 18.7727852, -67.8674164, 67.8105240
12: -31.1371765, 45.5780106, -31.3201733, 45.6127777, -76.7499542, 76.8981857
13: -29.4824753, 70.1800690, -29.7704945, 70.2113724, -99.6938477, 99.9505615
14: -67.4125977, 33.1243629, -67.4946899, 33.1563911, -100.5689850, 100.6190491
15: -35.4214554, 36.9216995, -35.4706192, 36.9831238, -72.4045792, 72.3923187
16: -54.9096794, 24.9401970, -54.9721565, 24.9970131, -79.9066925, 79.9123535
17: -55.4338684, 40.6690483, -55.5371628, 40.7147026, -96.1485748, 96.2062073
18: -60.4463158, 16.0997581, -60.4798927, 16.2512779, -76.6975937, 76.5796509
19: -42.9769058, 15.1190357, -43.0108299, 15.1954212, -58.1723251, 58.1298676
20: -40.2914696, 20.0084591, -40.3253746, 20.1531658, -60.4446335, 60.3338318
21: -51.9492073, 17.0044193, -51.9985962, 17.1575890, -69.1067963, 69.0030136
22: -41.6213493, 26.9690132, -41.6899681, 27.0544968, -68.6758423, 68.6589813
23: -41.4776306, 23.6932182, -41.5016098, 23.8018646, -65.2794952, 65.1948242
24: -55.3537979, 20.6183395, -55.3932915, 20.7463455, -76.1001434, 76.0116272
25: -36.4096184, 30.0189171, -36.4474487, 30.1448822, -66.5545044, 66.4663696
26: -56.7668152, 25.6511784, -56.8232727, 25.8179588, -82.5847778, 82.4744492
27: -66.7386780, 11.8918428, -66.7759857, 12.0809059, -78.8195801, 78.6678314
28: -41.2925873, 27.7180290, -41.3117371, 27.8177357, -69.1103210, 69.0297699
29: -42.6371994, 25.1322002, -42.7013435, 25.2032509, -67.8404541, 67.8335419
30: -51.1520271, 24.9613113, -51.1912918, 25.1476402, -76.2996674, 76.1526031
31: -57.9818001, 22.2124672, -58.0398598, 22.3212147, -80.3030167, 80.2523270
32: -44.9571075, 29.7127495, -45.0776443, 29.7318325, -74.6889420, 74.7903900
33: -75.8357697, 30.8840866, -75.9939346, 30.9112167, -106.7469864, 106.8780212
34: -62.0520058, 19.5380554, -62.0917282, 19.5777245, -81.6297302, 81.6297836
35: -56.7842484, 29.5820389, -56.9421654, 29.6099815, -86.3942261, 86.5242004
36: -54.8815117, 29.1012764, -55.0386658, 29.1259575, -84.0074692, 84.1399384
37: -94.3506622, 6.4652071, -94.5214081, 6.4856977, -100.8363571, 100.9866180
38: -71.4727249, 30.2960091, -71.6153412, 30.3340683, -101.8067932, 101.9113464
39: -79.9787750, 27.2653770, -80.1997299, 27.2812233, -107.2599945, 107.4651031
40: -79.6989899, 0.4665899, -79.7923279, 0.4922543, -80.1912460, 80.2589188
41: -57.1555519, 21.6219711, -57.2787285, 21.6471977, -78.8027496, 78.9006958
42: -35.8355942, 22.1245003, -35.8986969, 22.1536236, -57.9892197, 58.0231972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=475, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
time: 77.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
time: 78.60 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -62.0458031, 35.7048416, -62.0829086, 35.6934280, -97.7392273, 97.7877502
1: -26.3214149, 29.9184208, -26.3437309, 29.8986473, -56.2200623, 56.2621536
2: -26.3323364, 31.0064240, -26.3612347, 30.9743137, -57.3066483, 57.3676605
3: -26.1232986, 39.1286278, -26.1362457, 39.1061096, -65.2294083, 65.2648773
4: -36.4622040, 31.7457733, -36.5008698, 31.7365875, -68.1987915, 68.2466431
5: -27.7474747, 36.0658913, -27.7638397, 36.0580444, -63.8055191, 63.8297310
6: -56.4781494, 22.9946480, -56.4950027, 22.9316158, -79.4097672, 79.4896545
7: -35.5299911, 27.1622143, -35.5593491, 27.1553574, -62.6853485, 62.7215652
8: -47.2724380, 38.1720276, -47.2920723, 38.1956177, -85.4680557, 85.4640961
9: -31.4518032, 42.5420380, -31.4697647, 42.4994965, -73.9513016, 74.0118027
10: -45.5224762, 54.4056778, -45.5281410, 54.3153992, -99.8378754, 99.9338226
11: -49.1655579, 18.7133598, -49.1478653, 18.7795658, -67.9451218, 67.8612213
12: -31.2033272, 45.6965027, -31.3403950, 45.6236382, -76.8269653, 77.0368958
13: -29.6311131, 70.3887863, -29.8322716, 70.2185364, -99.8496475, 100.2210541
14: -67.5119247, 33.1779175, -67.5082016, 33.1691551, -100.6810760, 100.6861191
15: -35.5138702, 36.9883270, -35.4809952, 37.0055313, -72.5194016, 72.4693222
16: -54.9833641, 25.0730667, -54.9863434, 25.0055885, -79.9889526, 80.0594101
17: -55.5387955, 40.7332230, -55.5503693, 40.7275238, -96.2663193, 96.2835922
18: -60.6126556, 16.2540359, -60.4867325, 16.3152924, -76.9279480, 76.7407684
19: -43.0658875, 15.1696987, -43.0180397, 15.2133446, -58.2792320, 58.1877365
20: -40.3825989, 20.0672493, -40.3325539, 20.1772041, -60.5598030, 60.3998032
21: -52.0472984, 17.0566483, -52.0089111, 17.1772327, -69.2245331, 69.0655594
22: -41.8061333, 27.0608044, -41.6995354, 27.0927162, -68.8988495, 68.7603378
23: -41.5762939, 23.7492733, -41.5062218, 23.8237514, -65.4000473, 65.2554932
24: -55.4927292, 20.6933308, -55.3977127, 20.7740269, -76.2667542, 76.0910416
25: -36.5077438, 30.0823002, -36.4533806, 30.1692371, -66.6769791, 66.5356827
26: -56.9697418, 25.7975559, -56.8308868, 25.8798141, -82.8495560, 82.6284409
27: -66.9279251, 11.9986563, -66.7828369, 12.1277628, -79.0556870, 78.7814941
28: -41.4207268, 27.7978897, -41.3176041, 27.8521709, -69.2728958, 69.1154938
29: -42.7996254, 25.1932487, -42.7091255, 25.2281075, -68.0277328, 67.9023743
30: -51.2648087, 25.0218277, -51.1987724, 25.1702003, -76.4350128, 76.2205963
31: -58.0883369, 22.2729244, -58.0500565, 22.3412361, -80.4295731, 80.3229828
32: -45.0530624, 29.7976093, -45.1055946, 29.7353935, -74.7884521, 74.9032059
33: -75.9113235, 30.9828815, -76.0166168, 30.9211407, -106.8324661, 106.9994965
34: -62.1082268, 19.5899849, -62.1037216, 19.5856838, -81.6939087, 81.6937103
35: -56.8473663, 29.6647911, -56.9614868, 29.6163559, -86.4637222, 86.6262817
36: -54.9461784, 29.1497307, -55.0559044, 29.1305714, -84.0767517, 84.2056351
37: -94.4129410, 6.5256052, -94.5303650, 6.4942894, -100.9072266, 101.0559692
38: -71.5412445, 30.3871078, -71.6347961, 30.3407116, -101.8819580, 102.0219040
39: -80.1054001, 27.4242001, -80.2451248, 27.2894497, -107.3948517, 107.6693268
40: -79.7825470, 0.5392857, -79.8037643, 0.5003490, -80.2828979, 80.3430481
41: -57.2358170, 21.6915283, -57.3017120, 21.6530857, -78.8889008, 78.9932404
42: -35.9144211, 22.2176056, -35.9208145, 22.1601963, -58.0746155, 58.1384201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
time: 79.69 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
time: 78.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -62.0657806, 35.7406921, -61.9986267, 35.6004295, -97.6662140, 97.7393188
1: -26.3228035, 29.9354572, -26.2815056, 29.8375168, -56.1603203, 56.2169647
2: -26.3541260, 31.0310707, -26.3313446, 30.9291859, -57.2833099, 57.3624153
3: -26.1114655, 39.1790123, -26.0865612, 39.0468636, -65.1583252, 65.2655716
4: -36.5098076, 31.7963486, -36.4769821, 31.6838093, -68.1936188, 68.2733307
5: -27.7094841, 36.1154366, -27.6641617, 35.9867973, -63.6962814, 63.7795982
6: -56.4638901, 22.9513664, -56.4344482, 22.8921509, -79.3560410, 79.3858185
7: -35.4942513, 27.1996155, -35.4228668, 27.0882206, -62.5824738, 62.6224823
8: -47.3083420, 38.2325897, -47.2552872, 38.1108780, -85.4192200, 85.4878769
9: -31.4328117, 42.5323105, -31.3803558, 42.4500656, -73.8828735, 73.9126663
10: -45.4247665, 54.3031464, -45.3117218, 54.2568207, -99.6815872, 99.6148682
11: -49.1389771, 18.7488422, -48.9140129, 18.7165070, -67.8554840, 67.6628571
12: -31.1826382, 45.6092453, -31.1481876, 45.4651146, -76.6477509, 76.7574310
13: -29.6503487, 70.3458710, -29.6471443, 70.0260239, -99.6763763, 99.9930115
14: -67.5080566, 33.1945038, -67.4077377, 33.1272430, -100.6352997, 100.6022415
15: -35.4802399, 36.9566574, -35.4332047, 36.9427528, -72.4229889, 72.3898621
16: -54.9150009, 24.9941025, -54.8097420, 24.9356842, -79.8506851, 79.8038483
17: -55.5325279, 40.7800026, -55.4061279, 40.5485153, -96.0810394, 96.1861267
18: -60.5783463, 16.2170620, -60.3659286, 16.2273674, -76.8057098, 76.5829926
19: -43.0518875, 15.1937284, -42.8934975, 15.1859016, -58.2377892, 58.0872269
20: -40.3276443, 20.0725288, -40.1921806, 20.0616074, -60.3892517, 60.2647095
21: -52.0097504, 17.0983200, -51.8082275, 17.0827255, -69.0924759, 68.9065475
22: -41.7026062, 27.0362644, -41.5497513, 27.0388412, -68.7414474, 68.5860138
23: -41.5965042, 23.7917995, -41.4135742, 23.7740040, -65.3705063, 65.2053757
24: -55.4304428, 20.7044201, -55.2482758, 20.6948509, -76.1252899, 75.9526978
25: -36.4879608, 30.1189823, -36.3333206, 30.1031799, -66.5911407, 66.4523010
26: -56.9128876, 25.7541733, -56.7234268, 25.7720375, -82.6849213, 82.4776001
27: -66.8393478, 11.9967041, -66.6035004, 12.0078144, -78.8471603, 78.6002045
28: -41.4063873, 27.8053360, -41.2641449, 27.8038063, -69.2101898, 69.0694809
29: -42.6890411, 25.2063046, -42.4962196, 25.1999626, -67.8890076, 67.7025223
30: -51.1579819, 25.0410519, -50.9977264, 25.0197449, -76.1777267, 76.0387802
31: -58.0991936, 22.3187828, -57.8871269, 22.3016243, -80.4008179, 80.2059097
32: -45.0316505, 29.7320328, -45.0003510, 29.6325397, -74.6641922, 74.7323837
33: -75.9141388, 30.9287262, -75.8812866, 30.7453194, -106.6594543, 106.8100128
34: -62.1118011, 19.4889851, -62.0792046, 19.3732300, -81.4850311, 81.5681915
35: -56.8347778, 29.5396328, -56.8108635, 29.3824635, -86.2172394, 86.3504944
36: -54.9430618, 29.0504456, -54.9146385, 28.9365730, -83.8796387, 83.9650879
37: -94.4243469, 6.4323435, -94.3484650, 6.3121929, -100.7365417, 100.7808075
38: -71.5810089, 30.2257977, -71.5159149, 30.0740166, -101.6550293, 101.7417145
39: -80.1138611, 27.3131523, -80.0812454, 27.0749130, -107.1887741, 107.3943939
40: -79.7891388, 0.4973145, -79.7154999, 0.3818350, -80.1709747, 80.2128143
41: -57.2160263, 21.5984974, -57.1879234, 21.5435829, -78.7596130, 78.7864227
42: -35.8712883, 22.1582279, -35.8483810, 22.1280918, -57.9993820, 58.0066071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=475, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
time: 91.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
time: 85.06 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1346855, 35.7903252, -62.0113640, 35.6060333, -97.7407227, 97.8016891
1: -26.3699226, 29.9959602, -26.2986736, 29.8414993, -56.2114220, 56.2946320
2: -26.3980846, 31.0927792, -26.3463554, 30.9344845, -57.3325691, 57.4391327
3: -26.1830025, 39.2491302, -26.1080399, 39.0505142, -65.2335205, 65.3571701
4: -36.5527878, 31.8422985, -36.4876099, 31.6881714, -68.2409592, 68.3299103
5: -27.7636814, 36.1834908, -27.6807270, 35.9912949, -63.7549744, 63.8642197
6: -56.5321236, 23.0569439, -56.4520226, 22.8983498, -79.4304733, 79.5089645
7: -35.5588417, 27.2703857, -35.4457932, 27.0926971, -62.6515388, 62.7161789
8: -47.3602028, 38.2877388, -47.2644386, 38.1193810, -85.4795837, 85.5521774
9: -31.5004959, 42.6106644, -31.4014091, 42.4533691, -73.9538651, 74.0120697
10: -45.5227737, 54.4476624, -45.3443794, 54.2646866, -99.7874603, 99.7920380
11: -49.2099533, 18.7866058, -48.9268990, 18.7232857, -67.9332428, 67.7135010
12: -31.2487812, 45.7277489, -31.1683941, 45.4759140, -76.7246933, 76.8961411
13: -29.7989616, 70.5546036, -29.7089386, 70.0332108, -99.8321686, 100.2635422
14: -67.6072845, 33.2480469, -67.4211884, 33.1399803, -100.7472687, 100.6692352
15: -35.5726089, 37.0232697, -35.4435959, 36.9652290, -72.5378418, 72.4668655
16: -54.9886932, 25.1269894, -54.8238907, 24.9442368, -79.9329300, 79.9508820
17: -55.6374168, 40.8442612, -55.4193192, 40.5613785, -96.1987915, 96.2635803
18: -60.7446594, 16.3713379, -60.3727646, 16.2914028, -77.0360641, 76.7441025
19: -43.1408386, 15.2443752, -42.9007492, 15.2038383, -58.3446770, 58.1451263
20: -40.4188156, 20.1313171, -40.1993866, 20.0856705, -60.5044861, 60.3307037
21: -52.1078300, 17.1505146, -51.8185425, 17.1023560, -69.2101898, 68.9690552
22: -41.8873863, 27.1280308, -41.5593529, 27.0770473, -68.9644318, 68.6873856
23: -41.6951256, 23.8478565, -41.4181519, 23.7959042, -65.4910278, 65.2660065
24: -55.5693893, 20.7794266, -55.2527084, 20.7225800, -76.2919693, 76.0321350
25: -36.5860825, 30.1823921, -36.3393135, 30.1275330, -66.7136154, 66.5217056
26: -57.1158218, 25.9005394, -56.7310295, 25.8338890, -82.9497070, 82.6315689
27: -67.0286255, 12.1035309, -66.6103745, 12.0546827, -79.0833054, 78.7139053
28: -41.5345078, 27.8852062, -41.2700081, 27.8382187, -69.3727264, 69.1552124
29: -42.8515053, 25.2673111, -42.5040016, 25.2248154, -68.0763245, 67.7713165
30: -51.2708206, 25.1016006, -51.0052185, 25.0422649, -76.3130875, 76.1068192
31: -58.2057419, 22.3792496, -57.8973503, 22.3216381, -80.5273819, 80.2765961
32: -45.1275940, 29.8168831, -45.0282440, 29.6360779, -74.7636719, 74.8451233
33: -75.9896545, 31.0275612, -75.9039917, 30.7552681, -106.7449188, 106.9315491
34: -62.1680679, 19.5409431, -62.0911865, 19.3812485, -81.5493164, 81.6321259
35: -56.8979797, 29.6223831, -56.8301239, 29.3888874, -86.2868652, 86.4525070
36: -55.0077324, 29.0988693, -54.9318466, 28.9412479, -83.9489822, 84.0307159
37: -94.4866638, 6.4927244, -94.3573532, 6.3207636, -100.8074265, 100.8500748
38: -71.6495972, 30.3169460, -71.5353165, 30.0806675, -101.7302628, 101.8522644
39: -80.2404861, 27.4719734, -80.1266479, 27.0831032, -107.3235931, 107.5986176
40: -79.8727417, 0.5700369, -79.7269135, 0.3899698, -80.2627106, 80.2969513
41: -57.2963104, 21.6680756, -57.2108841, 21.5494938, -78.8458023, 78.8789597
42: -35.9500809, 22.2513428, -35.8705139, 22.1346302, -58.0847092, 58.1218567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
time: 81.23 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
time: 67.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -62.1108246, 35.7854004, -62.1146278, 35.6966820, -97.8075104, 97.9000244
1: -26.3615284, 29.9571419, -26.3564587, 29.9022884, -56.2638168, 56.3135986
2: -26.3746834, 31.0494156, -26.3823204, 30.9757652, -57.3504486, 57.4317360
3: -26.1454697, 39.1942139, -26.1531258, 39.1112366, -65.2567062, 65.3473358
4: -36.5309105, 31.8146152, -36.5350418, 31.7393837, -68.2702942, 68.3496552
5: -27.7721291, 36.1359177, -27.7784748, 36.0639954, -63.8361244, 63.9143906
6: -56.4775963, 22.9747200, -56.4943008, 22.9465790, -79.4241791, 79.4690247
7: -35.5775375, 27.2146301, -35.5768738, 27.1607666, -62.7383041, 62.7915039
8: -47.3445320, 38.2545013, -47.3303490, 38.1995010, -85.5440369, 85.5848541
9: -31.4731712, 42.5518036, -31.4743958, 42.5042191, -73.9773865, 74.0261993
10: -45.5285568, 54.3272476, -45.5159492, 54.3227310, -99.8512878, 99.8432007
11: -49.2618828, 18.7661972, -49.1432495, 18.8081512, -68.0700378, 67.9094467
12: -31.2106361, 45.7050934, -31.3407936, 45.6341171, -76.8447571, 77.0458832
13: -29.6856842, 70.4543533, -29.8620071, 70.2222137, -99.9078979, 100.3163605
14: -67.5613861, 33.2081490, -67.5332260, 33.1632385, -100.7246246, 100.7413788
15: -35.5064278, 36.9706497, -35.4961395, 36.9960213, -72.5024490, 72.4667892
16: -55.0008430, 25.0152855, -54.9884377, 25.0088215, -80.0096664, 80.0037231
17: -55.6214828, 40.8709297, -55.5955925, 40.7248955, -96.3463745, 96.4665222
18: -60.6423187, 16.2419796, -60.4912682, 16.3121681, -76.9544830, 76.7332458
19: -43.1137924, 15.2071609, -43.0185165, 15.2354317, -58.3492241, 58.2256775
20: -40.3993378, 20.0891685, -40.3331299, 20.1859531, -60.5852890, 60.4222984
21: -52.1112671, 17.1126595, -52.0073509, 17.2041512, -69.3154144, 69.1200104
22: -41.7656975, 27.0538216, -41.7035332, 27.0903244, -68.8560181, 68.7573547
23: -41.6435471, 23.8090992, -41.5072784, 23.8509426, -65.4944916, 65.3163757
24: -55.5050049, 20.7242279, -55.4014206, 20.7932434, -76.2982483, 76.1256485
25: -36.5456924, 30.1355782, -36.4552574, 30.1938820, -66.7395782, 66.5908356
26: -56.9599304, 25.7794952, -56.8382797, 25.8751087, -82.8350372, 82.6177750
27: -66.9340134, 12.0179539, -66.7850113, 12.1397343, -79.0737457, 78.8029633
28: -41.4342041, 27.8251419, -41.3185310, 27.8649044, -69.2991104, 69.1436768
29: -42.7907104, 25.2212029, -42.7132835, 25.2417316, -68.0324402, 67.9344864
30: -51.2614975, 25.0591354, -51.1975708, 25.1852341, -76.4467316, 76.2567062
31: -58.1752167, 22.3353329, -58.0521851, 22.3754311, -80.5506439, 80.3875198
32: -45.0534401, 29.7946873, -45.1076927, 29.7441368, -74.7975769, 74.9023819
33: -75.9367142, 31.0318794, -76.0258179, 30.9239941, -106.8607101, 107.0576935
34: -62.1236725, 19.5984364, -62.1114807, 19.5846634, -81.7083359, 81.7099152
35: -56.8523865, 29.6774120, -56.9621544, 29.6214314, -86.4738159, 86.6395645
36: -54.9586067, 29.1620617, -55.0623093, 29.1344490, -84.0930557, 84.2243729
37: -94.4616394, 6.5404987, -94.5495453, 6.4966145, -100.9582520, 101.0900421
38: -71.5997772, 30.3735466, -71.6464996, 30.3438416, -101.9436188, 102.0200500
39: -80.1430817, 27.4363461, -80.2604980, 27.2862740, -107.4293518, 107.6968460
40: -79.8118744, 0.5632143, -79.8205109, 0.5012236, -80.3130951, 80.3837280
41: -57.2294579, 21.6609459, -57.2978668, 21.6590805, -78.8885345, 78.9588165
42: -35.8863068, 22.1777515, -35.9109039, 22.1704216, -58.0567284, 58.0886536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
time: 81.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
time: 77.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -62.1797562, 35.8350067, -62.1273804, 35.7022896, -97.8820496, 97.9623871
1: -26.4086361, 30.0176182, -26.3736458, 29.9062424, -56.3148804, 56.3912659
2: -26.4186401, 31.1110973, -26.3973045, 30.9810390, -57.3996811, 57.5084000
3: -26.2170639, 39.2643814, -26.1746407, 39.1149216, -65.3319855, 65.4390259
4: -36.5738945, 31.8606052, -36.5456429, 31.7438107, -68.3177032, 68.4062500
5: -27.8263321, 36.2039871, -27.7950325, 36.0685349, -63.8948669, 63.9990196
6: -56.5458298, 23.0802860, -56.5118790, 22.9527225, -79.4985504, 79.5921631
7: -35.6420708, 27.2854614, -35.5997963, 27.1652908, -62.8073616, 62.8852577
8: -47.3963699, 38.3097153, -47.3395081, 38.2080154, -85.6043854, 85.6492233
9: -31.5408249, 42.6301308, -31.4954338, 42.5075073, -74.0483322, 74.1255646
10: -45.6265564, 54.4717331, -45.5486374, 54.3305664, -99.9571228, 100.0203705
11: -49.3329277, 18.8039913, -49.1561699, 18.8149300, -68.1478577, 67.9601593
12: -31.2767506, 45.8235931, -31.3610210, 45.6449356, -76.9216843, 77.1846161
13: -29.8343067, 70.6630707, -29.9237480, 70.2293472, -100.0636520, 100.5868225
14: -67.6606445, 33.2617569, -67.5467682, 33.1760292, -100.8366699, 100.8085251
15: -35.5988617, 37.0373192, -35.5065498, 37.0184135, -72.6172791, 72.5438690
16: -55.0744667, 25.1481934, -55.0026321, 25.0173607, -80.0918274, 80.1508255
17: -55.7264519, 40.9350967, -55.6087761, 40.7377243, -96.4641724, 96.5438690
18: -60.8086624, 16.3962364, -60.4980583, 16.3761940, -77.1848602, 76.8942947
19: -43.2027588, 15.2578154, -43.0257339, 15.2533550, -58.4561157, 58.2835503
20: -40.4904785, 20.1479435, -40.3403320, 20.2100143, -60.7004929, 60.4882736
21: -52.2094154, 17.1648693, -52.0176468, 17.2238235, -69.4332428, 69.1825180
22: -41.9504929, 27.1455784, -41.7131157, 27.1285686, -69.0790634, 68.8586960
23: -41.7421951, 23.8651600, -41.5118713, 23.8728447, -65.6150360, 65.3770294
24: -55.6439590, 20.7991905, -55.4058304, 20.8209839, -76.4649429, 76.2050171
25: -36.6438484, 30.1989326, -36.4612236, 30.2182217, -66.8620682, 66.6601562
26: -57.1628342, 25.9258842, -56.8459206, 25.9369564, -83.0997925, 82.7718048
27: -67.1232605, 12.1247864, -66.7918549, 12.1865959, -79.3098602, 78.9166412
28: -41.5623474, 27.9049683, -41.3243866, 27.8993111, -69.4616547, 69.2293549
29: -42.9531364, 25.2822075, -42.7210579, 25.2666130, -68.2197495, 68.0032654
30: -51.3743210, 25.1196613, -51.2050629, 25.2077789, -76.5820999, 76.3247223
31: -58.2817955, 22.3957825, -58.0623817, 22.3954563, -80.6772537, 80.4581604
32: -45.1493492, 29.8795547, -45.1355896, 29.7476711, -74.8970184, 75.0151443
33: -76.0123138, 31.1307411, -76.0484772, 30.9339466, -106.9462585, 107.1792145
34: -62.1799850, 19.6503716, -62.1234322, 19.5926781, -81.7726593, 81.7738037
35: -56.9155502, 29.7602005, -56.9814796, 29.6278000, -86.5433502, 86.7416840
36: -55.0233002, 29.2104778, -55.0795403, 29.1390915, -84.1623917, 84.2900162
37: -94.5239563, 6.6009073, -94.5584869, 6.5051565, -101.0291138, 101.1593933
38: -71.6684265, 30.4647160, -71.6658783, 30.3504963, -102.0189209, 102.1305923
39: -80.2697067, 27.5951805, -80.3059158, 27.2945404, -107.5642471, 107.9010925
40: -79.8954773, 0.6359892, -79.8319321, 0.5093203, -80.4048004, 80.4679184
41: -57.3097687, 21.7305126, -57.3208008, 21.6649818, -78.9747467, 79.0513153
42: -35.9651260, 22.2708664, -35.9330292, 22.1769505, -58.1420746, 58.2038956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1677

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
time: 83.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
time: 69.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 154.85 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6933544
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.6953026
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.85
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -61.9239731, 35.6056061, -62.0104332, 35.6264267, -97.5503998, 97.6160431
1: -26.2284126, 29.8335114, -26.2645626, 29.8663750, -56.0947876, 56.0980759
2: -26.2635880, 30.9237385, -26.3202419, 30.9604530, -57.2240410, 57.2439804
3: -26.0095711, 39.0396881, -26.0893040, 39.1131668, -65.1227417, 65.1289902
4: -36.3943558, 31.6695156, -36.5035172, 31.7027454, -68.0970993, 68.1730347
5: -27.6242065, 35.9746323, -27.6649399, 36.0431404, -63.6673470, 63.6395721
6: -56.3882217, 22.8631706, -56.4578552, 23.0174828, -79.4057007, 79.3210297
7: -35.3740158, 27.0737724, -35.4056015, 27.1395988, -62.5136147, 62.4793739
8: -47.1821671, 38.0896759, -47.2694550, 38.1241341, -85.3063049, 85.3591309
9: -31.3388519, 42.4409828, -31.3846684, 42.5130577, -73.8519135, 73.8256531
10: -45.3123665, 54.2321854, -45.3370857, 54.4016647, -99.7140350, 99.5692749
11: -48.9608841, 18.6545086, -48.9510498, 18.7374630, -67.6983490, 67.6055603
12: -31.1011276, 45.4789848, -31.1520786, 45.5973206, -76.6984482, 76.6310654
13: -29.4341354, 70.0676346, -29.5800514, 70.2190170, -99.6531525, 99.6476898
14: -67.3528290, 33.0974655, -67.4645767, 33.1422844, -100.4951172, 100.5620422
15: -35.3918381, 36.9011078, -35.5225410, 36.9571457, -72.3489838, 72.4236450
16: -54.8075294, 24.9149723, -54.8320351, 25.0695953, -79.8771210, 79.7470093
17: -55.3364639, 40.5683746, -55.4389420, 40.5801811, -95.9166412, 96.0073166
18: -60.3779526, 16.0646877, -60.5461273, 16.2052402, -76.5831909, 76.6108170
19: -42.9111214, 15.1009655, -42.9777870, 15.1642046, -58.0753250, 58.0787506
20: -40.2163811, 19.9861279, -40.2763596, 20.0456543, -60.2620354, 60.2624893
21: -51.8421021, 16.9857006, -51.8836212, 17.0587349, -68.9008331, 68.8693237
22: -41.5528603, 26.9437981, -41.7271652, 27.0249023, -68.5777588, 68.6709595
23: -41.4274559, 23.6702080, -41.5095596, 23.7411327, -65.1685867, 65.1797638
24: -55.2749367, 20.5905151, -55.3909416, 20.6681118, -75.9430466, 75.9814606
25: -36.3489838, 29.9941673, -36.4195671, 30.0695362, -66.4185181, 66.4137344
26: -56.7156296, 25.6123314, -56.9253616, 25.7471046, -82.4627380, 82.5376892
27: -66.6396332, 11.8614750, -66.7933426, 11.9696026, -78.6092377, 78.6548157
28: -41.2616577, 27.6904068, -41.3864784, 27.7691612, -69.0308228, 69.0768890
29: -42.5288353, 25.1113663, -42.6348648, 25.1746712, -67.7035065, 67.7462311
30: -51.0437698, 24.9374886, -51.0814972, 25.0120068, -76.0557785, 76.0189819
31: -57.9010773, 22.1898727, -57.9786949, 22.2694340, -80.1705093, 80.1685638
32: -44.9235992, 29.6484909, -45.0108566, 29.7198524, -74.6434479, 74.6593475
33: -75.8034744, 30.7765465, -75.8761215, 30.8353062, -106.6387787, 106.6526642
34: -62.0316467, 19.4244499, -62.0915298, 19.4431610, -81.4748077, 81.5159760
35: -56.7569275, 29.4404526, -56.8137054, 29.4647942, -86.2217255, 86.2541580
36: -54.8589706, 28.9876919, -54.9190903, 28.9776058, -83.8365784, 83.9067841
37: -94.3076019, 6.3530579, -94.3696594, 6.3806133, -100.6882172, 100.7227173
38: -71.4464264, 30.1441345, -71.5163345, 30.1438599, -101.5902863, 101.6604691
39: -79.9394379, 27.1378727, -80.0608521, 27.2196217, -107.1590576, 107.1987228
40: -79.6718750, 0.3956995, -79.7489319, 0.4565287, -80.1284027, 80.1446304
41: -57.1341858, 21.5573616, -57.2085266, 21.6120987, -78.7462845, 78.7658844
42: -35.8130035, 22.1011448, -35.8788071, 22.2447643, -58.0577698, 57.9799500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=473, inp2_unstable=474, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6798585
time: 73.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7414276
time: 68.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -61.9926720, 35.6551323, -62.0230217, 35.6320381, -97.6247101, 97.6781540
1: -26.2754993, 29.8939571, -26.2814808, 29.8702774, -56.1457748, 56.1754379
2: -26.3075390, 30.9853172, -26.3350163, 30.9656563, -57.2731934, 57.3203354
3: -26.0805016, 39.1096649, -26.1106262, 39.1167488, -65.1972504, 65.2202911
4: -36.4371033, 31.7152710, -36.5137787, 31.7069950, -68.1440964, 68.2290497
5: -27.6781235, 36.0425034, -27.6812191, 36.0475235, -63.7256470, 63.7237244
6: -56.4562378, 22.9684734, -56.4751816, 23.0233383, -79.4795761, 79.4436569
7: -35.4383659, 27.1444416, -35.4281998, 27.1440430, -62.5824089, 62.5726395
8: -47.2338257, 38.1447067, -47.2781448, 38.1324959, -85.3663177, 85.4228516
9: -31.4062138, 42.5189896, -31.4054432, 42.5159492, -73.9221649, 73.9244308
10: -45.4101868, 54.3765106, -45.3695602, 54.4093208, -99.8195038, 99.7460709
11: -49.0313797, 18.6922207, -48.9630165, 18.7439957, -67.7753754, 67.6552353
12: -31.1669827, 45.5972672, -31.1720524, 45.6078491, -76.7748337, 76.7693176
13: -29.5826721, 70.2762299, -29.6417599, 70.2260284, -99.8087006, 99.9179916
14: -67.4517670, 33.1510201, -67.4778595, 33.1549072, -100.6066742, 100.6288757
15: -35.4839668, 36.9668961, -35.5325050, 36.9786835, -72.4626465, 72.4994049
16: -54.8827972, 25.0478172, -54.8467827, 25.0779934, -79.9607925, 79.8945999
17: -55.4410133, 40.6324692, -55.4516678, 40.5926170, -96.0336304, 96.0841370
18: -60.5441132, 16.2175789, -60.5528336, 16.2686539, -76.8127670, 76.7704163
19: -42.9999504, 15.1515408, -42.9848328, 15.1819944, -58.1819458, 58.1363754
20: -40.3074265, 20.0446091, -40.2834549, 20.0694084, -60.3768349, 60.3280640
21: -51.9400673, 17.0377007, -51.8937416, 17.0781288, -69.0181961, 68.9314423
22: -41.7375069, 27.0350361, -41.7366180, 27.0626202, -68.8001251, 68.7716522
23: -41.5260162, 23.7261581, -41.5140495, 23.7628288, -65.2888489, 65.2402039
24: -55.4136696, 20.6648636, -55.3951797, 20.6955929, -76.1092606, 76.0600433
25: -36.4469910, 30.0573921, -36.4254379, 30.0936127, -66.5406036, 66.4828339
26: -56.9184303, 25.7585106, -56.9328461, 25.8088608, -82.7272949, 82.6913605
27: -66.8287354, 11.9681416, -66.8000336, 12.0163631, -78.8451004, 78.7681732
28: -41.3896408, 27.7700310, -41.3921814, 27.8034019, -69.1930389, 69.1622162
29: -42.6911812, 25.1722374, -42.6425362, 25.1994228, -67.8906021, 67.8147736
30: -51.1564026, 24.9977150, -51.0887375, 25.0341530, -76.1905518, 76.0864563
31: -58.0074387, 22.2497654, -57.9886475, 22.2885780, -80.2960205, 80.2384109
32: -45.0189323, 29.7332153, -45.0380936, 29.7231770, -74.7421112, 74.7713089
33: -75.8787460, 30.8751812, -75.8985901, 30.8449249, -106.7236710, 106.7737732
34: -62.0876465, 19.4764252, -62.1033669, 19.4511452, -81.5387878, 81.5797882
35: -56.8205986, 29.5232639, -56.8332558, 29.4710369, -86.2916336, 86.3565216
36: -54.9237556, 29.0361404, -54.9361687, 28.9822006, -83.9059601, 83.9723053
37: -94.3695526, 6.4133301, -94.3783417, 6.3892155, -100.7587662, 100.7916718
38: -71.5146713, 30.2352619, -71.5355988, 30.1504192, -101.6650925, 101.7708588
39: -80.0658417, 27.2966061, -80.1058655, 27.2276497, -107.2934875, 107.4024734
40: -79.7552032, 0.4684696, -79.7600555, 0.4645367, -80.2197418, 80.2285233
41: -57.2139893, 21.6268501, -57.2310753, 21.6179447, -78.8319321, 78.8579254
42: -35.8915253, 22.1940880, -35.9007072, 22.2510357, -58.1425629, 58.0947952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=474, inp2_unstable=475, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6798585
time: 65.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7414276
time: 72.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -61.9691010, 35.6502800, -62.1265411, 35.7227631, -97.6918640, 97.7768250
1: -26.2671528, 29.8551865, -26.3395081, 29.9311562, -56.1983109, 56.1946945
2: -26.2841511, 30.9420471, -26.3711891, 31.0070763, -57.2912292, 57.3132362
3: -26.0435772, 39.0548553, -26.1558590, 39.1775589, -65.2211380, 65.2107162
4: -36.4154587, 31.6877899, -36.5615921, 31.7583656, -68.1738281, 68.2493820
5: -27.6868610, 35.9950638, -27.7792950, 36.1201973, -63.8070602, 63.7743607
6: -56.4018860, 22.8865166, -56.5176926, 23.0719490, -79.4738312, 79.4042053
7: -35.4573364, 27.0887585, -35.5596237, 27.2121086, -62.6694450, 62.6483841
8: -47.2183533, 38.1115913, -47.3445282, 38.2126541, -85.4310074, 85.4561157
9: -31.3792191, 42.4603729, -31.4787540, 42.5672073, -73.9464264, 73.9391251
10: -45.4161949, 54.2562447, -45.5413437, 54.4675140, -99.8837128, 99.7975922
11: -49.0837135, 18.6718941, -49.1802902, 18.8290901, -67.9128036, 67.8521881
12: -31.1290874, 45.5748978, -31.3445721, 45.7663498, -76.8954391, 76.9194717
13: -29.4695320, 70.1761703, -29.7948761, 70.4151688, -99.8847046, 99.9710464
14: -67.4062576, 33.1111107, -67.5901794, 33.1782990, -100.5845566, 100.7012939
15: -35.4180298, 36.9151077, -35.5854874, 37.0104294, -72.4284592, 72.5005951
16: -54.8933563, 24.9360790, -55.0106812, 25.1427059, -80.0360641, 79.9467621
17: -55.4253120, 40.6592484, -55.6285629, 40.7566223, -96.1819305, 96.2878113
18: -60.4418793, 16.0896378, -60.6714478, 16.2899513, -76.7318268, 76.7610855
19: -42.9730148, 15.1144161, -43.1027603, 15.2136688, -58.1866837, 58.2171783
20: -40.2880783, 20.0027962, -40.4173088, 20.1699390, -60.4580154, 60.4201050
21: -51.9436646, 17.0000687, -52.0826912, 17.1801987, -69.1238632, 69.0827637
22: -41.6159668, 26.9613419, -41.8810768, 27.0763836, -68.6923523, 68.8424225
23: -41.4744987, 23.6875763, -41.6032372, 23.8179970, -65.2924957, 65.2908173
24: -55.3495178, 20.6103153, -55.5440598, 20.7664490, -76.1159668, 76.1543732
25: -36.4067307, 30.0107632, -36.5414963, 30.1602421, -66.5669708, 66.5522614
26: -56.7626724, 25.6376305, -57.0402222, 25.8501530, -82.6128235, 82.6778564
27: -66.7342682, 11.8827763, -66.9748535, 12.1014681, -78.8357391, 78.8576279
28: -41.2894669, 27.7101765, -41.4408150, 27.8301945, -69.1196594, 69.1509933
29: -42.6304283, 25.1262989, -42.8519897, 25.2164268, -67.8468552, 67.9782867
30: -51.1472626, 24.9555740, -51.2813148, 25.1774330, -76.3246918, 76.2368927
31: -57.9771385, 22.2064590, -58.1437416, 22.3430862, -80.3202209, 80.3502045
32: -44.9453850, 29.7112083, -45.1181145, 29.8314629, -74.7768478, 74.8293228
33: -75.8261261, 30.8797455, -76.0205383, 31.0139389, -106.8400650, 106.9002838
34: -62.0435257, 19.5339127, -62.1235504, 19.6546078, -81.6981354, 81.6574631
35: -56.7745361, 29.5782452, -56.9648438, 29.7036591, -86.4781952, 86.5430908
36: -54.8745117, 29.0993214, -55.0667496, 29.1754189, -84.0499268, 84.1660690
37: -94.3448792, 6.4612093, -94.5706482, 6.5650091, -100.9098892, 101.0318604
38: -71.4651642, 30.2918835, -71.6468353, 30.4137039, -101.8788681, 101.9387207
39: -79.9687500, 27.2611008, -80.2399445, 27.4309998, -107.3997498, 107.5010452
40: -79.6946106, 0.4616432, -79.8538361, 0.5759182, -80.2705307, 80.3154755
41: -57.1476212, 21.6198463, -57.3184509, 21.7275925, -78.8752136, 78.9382935
42: -35.8280334, 22.1206398, -35.9412766, 22.2870922, -58.1151276, 58.0619164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=474, inp2_unstable=475, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6821062
time: 70.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7434321
time: 75.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.0377541, 35.6998672, -62.1391335, 35.7283440, -97.7660980, 97.8390045
1: -26.3142300, 29.9156189, -26.3564281, 29.9350662, -56.2492981, 56.2720490
2: -26.3280888, 31.0036488, -26.3859692, 31.0122528, -57.3403397, 57.3896179
3: -26.1145515, 39.1248550, -26.1772556, 39.1811295, -65.2956848, 65.3021088
4: -36.4582024, 31.7335491, -36.5718307, 31.7625809, -68.2207794, 68.3053818
5: -27.7408142, 36.0629463, -27.7955723, 36.1246033, -63.8654175, 63.8585205
6: -56.4699554, 22.9918022, -56.5350075, 23.0777779, -79.5477295, 79.5268097
7: -35.5216599, 27.1594582, -35.5822067, 27.2165489, -62.7382088, 62.7416649
8: -47.2700081, 38.1665878, -47.3531876, 38.2210808, -85.4910889, 85.5197754
9: -31.4465561, 42.5383415, -31.4994926, 42.5701599, -74.0167160, 74.0378342
10: -45.5140152, 54.4005661, -45.5737572, 54.4751892, -99.9892044, 99.9743195
11: -49.1542625, 18.7096157, -49.1922302, 18.8356686, -67.9899292, 67.9018478
12: -31.1949940, 45.6931686, -31.3646259, 45.7768402, -76.9718323, 77.0577927
13: -29.6180439, 70.3847198, -29.8565636, 70.4221573, -100.0401993, 100.2412872
14: -67.5052414, 33.1646614, -67.6034470, 33.1909218, -100.6961670, 100.7681122
15: -35.5102081, 36.9809265, -35.5954819, 37.0319977, -72.5422058, 72.5764084
16: -54.9685898, 25.0689316, -55.0254211, 25.1511536, -80.1197433, 80.0943527
17: -55.5300560, 40.7234077, -55.6412773, 40.7689781, -96.2990341, 96.3646851
18: -60.6080704, 16.2424564, -60.6781807, 16.3533039, -76.9613724, 76.9206390
19: -43.0618286, 15.1649733, -43.1098289, 15.2314644, -58.2932930, 58.2748032
20: -40.3791008, 20.0612831, -40.4243851, 20.1937218, -60.5728226, 60.4856682
21: -52.0415916, 17.0520458, -52.0928345, 17.1995831, -69.2411728, 69.1448822
22: -41.8006401, 27.0525894, -41.8905258, 27.1141281, -68.9147644, 68.9431152
23: -41.5730324, 23.7434769, -41.6077576, 23.8397083, -65.4127426, 65.3512344
24: -55.4882889, 20.6847038, -55.5483093, 20.7939129, -76.2822037, 76.2330170
25: -36.5047302, 30.0739975, -36.5473175, 30.1842842, -66.6890106, 66.6213150
26: -56.9654388, 25.7838593, -57.0477295, 25.9118538, -82.8772888, 82.8315887
27: -66.9233551, 11.9893990, -66.9815521, 12.1482735, -79.0716248, 78.9709473
28: -41.4174576, 27.7898331, -41.4465408, 27.8644505, -69.2819061, 69.2363739
29: -42.7928009, 25.1871529, -42.8596497, 25.2411919, -68.0339966, 68.0467987
30: -51.2599258, 25.0158482, -51.2885399, 25.1996174, -76.4595413, 76.3043900
31: -58.0834503, 22.2662926, -58.1537247, 22.3622475, -80.4456940, 80.4200134
32: -45.0406761, 29.7958755, -45.1453781, 29.8347778, -74.8754578, 74.9412537
33: -75.9013977, 30.9783611, -76.0430527, 31.0236320, -106.9250336, 107.0214157
34: -62.0995750, 19.5858498, -62.1354218, 19.6625290, -81.7621002, 81.7212677
35: -56.8381691, 29.6610298, -56.9844437, 29.7099247, -86.5480957, 86.6454773
36: -54.9392891, 29.1477242, -55.0838737, 29.1800385, -84.1193237, 84.2315979
37: -94.4068146, 6.5214901, -94.5793686, 6.5736885, -100.9804993, 101.1008606
38: -71.5334167, 30.3830624, -71.6661072, 30.4202328, -101.9536514, 102.0491714
39: -80.0951462, 27.4198513, -80.2850342, 27.4390163, -107.5341644, 107.7048874
40: -79.7779846, 0.5344982, -79.8649979, 0.5839033, -80.3618851, 80.3994980
41: -57.2274246, 21.6893501, -57.3410263, 21.7334518, -78.9608765, 79.0303802
42: -35.9065666, 22.2136097, -35.9631958, 22.2933598, -58.1999283, 58.1768036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=475, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1677

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6821062
time: 314.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7434321
time: 73.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 390.62 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6798585
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7414276
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6798585
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7414276
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6821062
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7434321
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.6821062
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 390.62
Output dim: 13, lower bound: -55.5696688, upper bound: 55.7434321
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 390.62
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 390.62
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7431083
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 390.62
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 390.62
Output dim: 13, lower bound: -55.5715197, upper bound: 55.7450601

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 89.83 + 3585.14 = 3674.97 seconds
