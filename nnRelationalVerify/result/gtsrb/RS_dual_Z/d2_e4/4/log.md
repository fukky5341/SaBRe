## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.55 + 101.59 = 104.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -55.7676578, upper bound: 55.7676578

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964
time: 78.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7559964, upper bound: 55.6867869
time: 84.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 163.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 163.47
Output dim: 13, lower bound: -55.6867869, upper bound: 55.7559964
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 163.47
Output dim: 13, lower bound: -55.7559964, upper bound: 55.6867869

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6116438, upper bound: 55.7531827
time: 79.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6839434, upper bound: 55.6810154
time: 80.05 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6810154, upper bound: 55.6839434
time: 88.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7531827, upper bound: 55.6116438
time: 95.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 186.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 186.32
Output dim: 13, lower bound: -55.6116438, upper bound: 55.7531827
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 186.32
Output dim: 13, lower bound: -55.6839434, upper bound: 55.6810154
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 186.32
Output dim: 13, lower bound: -55.6810154, upper bound: 55.6839434
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 186.32
Output dim: 13, lower bound: -55.7531827, upper bound: 55.6116438

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5285750, upper bound: 55.7495126
time: 108.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6079372, upper bound: 55.6702158
time: 79.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6009777, upper bound: 55.6773126
time: 103.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6802645, upper bound: 55.5979520
time: 91.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5979520, upper bound: 55.6802645
time: 90.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6773126, upper bound: 55.6009778
time: 74.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6702158, upper bound: 55.6079372
time: 95.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7495126, upper bound: 55.5285750
time: 137.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 236.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.5285750, upper bound: 55.7495126
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.6079372, upper bound: 55.6702158
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.6009777, upper bound: 55.6773126
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.6802645, upper bound: 55.5979520
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.5979520, upper bound: 55.6802645
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.6773126, upper bound: 55.6009778
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.6702158, upper bound: 55.6079372
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 236.09
Output dim: 13, lower bound: -55.7495126, upper bound: 55.5285750

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.4741250, upper bound: 55.7469046
time: 81.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5257761, upper bound: 55.6956407
time: 83.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5535595, upper bound: 55.6676036
time: 79.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6051594, upper bound: 55.6163171
time: 97.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5466550, upper bound: 55.6747063
time: 83.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5982120, upper bound: 55.6233242
time: 90.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6259849, upper bound: 55.5953446
time: 89.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6774976, upper bound: 55.5439129
time: 80.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5439129, upper bound: 55.6774976
time: 91.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5953446, upper bound: 55.6259849
time: 88.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6233242, upper bound: 55.5982120
time: 96.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6747063, upper bound: 55.5466550
time: 86.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6163171, upper bound: 55.6051594
time: 93.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6676036, upper bound: 55.5535595
time: 95.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6956407, upper bound: 55.5257761
time: 86.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7469046, upper bound: 55.4741250
time: 127.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 216.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.4741250, upper bound: 55.7469046
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5257761, upper bound: 55.6956407
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5535595, upper bound: 55.6676036
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6051594, upper bound: 55.6163171
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5466550, upper bound: 55.6747063
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5982120, upper bound: 55.6233242
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6259849, upper bound: 55.5953446
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6774976, upper bound: 55.5439129
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5439129, upper bound: 55.6774976
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.5953446, upper bound: 55.6259849
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6233242, upper bound: 55.5982120
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6747063, upper bound: 55.5466550
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6163171, upper bound: 55.6051594
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6676036, upper bound: 55.5535595
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.6956407, upper bound: 55.5257761
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.90
Output dim: 13, lower bound: -55.7469046, upper bound: 55.4741250

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.4724899, upper bound: 55.6860074
time: 121.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.4130497, upper bound: 55.7452633
time: 147.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5241371, upper bound: 55.6347332
time: 96.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.4647052, upper bound: 55.6940268
time: 111.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5519988, upper bound: 55.6066202
time: 104.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.4925717, upper bound: 55.6659052
time: 78.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -62.1186981, 35.6952896, -62.1186981, 35.6952896, -97.8139877, 97.8139877
1: -26.3815155, 29.8916988, -26.3815155, 29.8916988, -56.2732162, 56.2732162
2: -26.3989525, 30.9784336, -26.3989525, 30.9784336, -57.3773880, 57.3773880
3: -26.1880188, 39.1026917, -26.1880188, 39.1026917, -65.2907104, 65.2907104
4: -36.5427551, 31.7350960, -36.5427551, 31.7350960, -68.2778473, 68.2778473
5: -27.8046131, 36.0353584, -27.8046131, 36.0353584, -63.8399734, 63.8399734
6: -56.4970093, 22.9477615, -56.4970093, 22.9477615, -79.4447708, 79.4447708
7: -35.6054230, 27.1279945, -35.6054230, 27.1279945, -62.7334175, 62.7334175
8: -47.3453407, 38.1683502, -47.3453407, 38.1683502, -85.5136871, 85.5136871
9: -31.4980659, 42.4946327, -31.4980659, 42.4946327, -73.9926987, 73.9926987
10: -45.5588722, 54.3206558, -45.5588722, 54.3206558, -99.8795319, 99.8795319
11: -49.1590309, 18.7700634, -49.1590309, 18.7700634, -67.9290924, 67.9290924
12: -31.2345753, 45.6529083, -31.2345753, 45.6529083, -76.8874817, 76.8874817
13: -29.8046360, 70.2310486, -29.8046360, 70.2310486, -100.0356827, 100.0356827
14: -67.5379791, 33.1873360, -67.5379791, 33.1873360, -100.7253113, 100.7253113
15: -35.5030632, 37.0071106, -35.5030632, 37.0071106, -72.5101776, 72.5101776
16: -55.0140762, 24.9919338, -55.0140762, 24.9919338, -80.0060120, 80.0060120
17: -55.5988579, 40.7363892, -55.5988579, 40.7363892, -96.3352509, 96.3352509
18: -60.4986916, 16.3675880, -60.4986916, 16.3675880, -76.8662796, 76.8662796
19: -43.0222435, 15.2431841, -43.0222435, 15.2431841, -58.2654266, 58.2654266
20: -40.3337784, 20.1320248, -40.3337784, 20.1320248, -60.4658051, 60.4658051
21: -52.0075035, 17.1464996, -52.0075035, 17.1464996, -69.1540070, 69.1540070
22: -41.6830597, 27.1268272, -41.6830597, 27.1268272, -68.8098907, 68.8098907
23: -41.5122375, 23.8434162, -41.5122375, 23.8434162, -65.3556519, 65.3556519
24: -55.3937492, 20.7823029, -55.3937492, 20.7823029, -76.1760559, 76.1760559
25: -36.4486122, 30.1759682, -36.4486122, 30.1759682, -66.6245804, 66.6245804
26: -56.8271561, 25.9055443, -56.8271561, 25.9055443, -82.7326965, 82.7326965
27: -66.7874069, 12.1168690, -66.7874069, 12.1168690, -78.9042740, 78.9042740
28: -41.3287277, 27.8905296, -41.3287277, 27.8905296, -69.2192535, 69.2192535
29: -42.6967773, 25.2648621, -42.6967773, 25.2648621, -67.9616394, 67.9616394
30: -51.1946220, 25.0940342, -51.1946220, 25.0940342, -76.2886581, 76.2886581
31: -58.0468330, 22.3735447, -58.0468330, 22.3735447, -80.4203796, 80.4203796
32: -45.0860748, 29.7509003, -45.0860748, 29.7509003, -74.8369751, 74.8369751
33: -75.9644623, 30.9451447, -75.9644623, 30.9451447, -106.9096069, 106.9096069
34: -62.1285019, 19.5887184, -62.1285019, 19.5887184, -81.7172241, 81.7172241
35: -56.8809738, 29.6318703, -56.8809738, 29.6318703, -86.5128479, 86.5128479
36: -54.9806023, 29.1390553, -54.9806023, 29.1390553, -84.1196594, 84.1196594
37: -94.4401779, 6.5176687, -94.4401779, 6.5176687, -100.9578476, 100.9578476
38: -71.5902786, 30.3442955, -71.5902786, 30.3442955, -101.9345703, 101.9345703
39: -80.2023926, 27.3069344, -80.2023926, 27.3069344, -107.5093231, 107.5093231
40: -79.8015289, 0.5129414, -79.8015289, 0.5129414, -80.3144684, 80.3144684
41: -57.2522469, 21.6652565, -57.2522469, 21.6652565, -78.9175034, 78.9175034
42: -35.9204178, 22.1770267, -35.9204178, 22.1770267, -58.0974426, 58.0974426

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6035996, upper bound: 55.5552712
time: 96.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.5442023, upper bound: 55.6146488
time: 84.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 182.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.4724899, upper bound: 55.6860074
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.4130497, upper bound: 55.7452633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.5241371, upper bound: 55.6347332
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.4647052, upper bound: 55.6940268
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.5519988, upper bound: 55.6066202
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.4925717, upper bound: 55.6659052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.6035996, upper bound: 55.5552712
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 182.35
Output dim: 13, lower bound: -55.5442023, upper bound: 55.6146488
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.5466550, upper bound: 55.6747063
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.5982120, upper bound: 55.6233242
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6259849, upper bound: 55.5953446
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6774976, upper bound: 55.5439129
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.5439129, upper bound: 55.6774976
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.5953446, upper bound: 55.6259849
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6233242, upper bound: 55.5982120
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6747063, upper bound: 55.5466550
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6163171, upper bound: 55.6051594
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6676036, upper bound: 55.5535595
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.6956407, upper bound: 55.5257761
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 182.35
Output dim: 13, lower bound: -55.7469046, upper bound: 55.4741250

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 104.15 + 3621.91 = 3726.05 seconds
