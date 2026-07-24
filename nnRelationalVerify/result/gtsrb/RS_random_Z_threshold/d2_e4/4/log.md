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
execution time: IAR + RelationalAnalysis = 2.78 + 86.23 = 89.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -55.7676578, upper bound: 55.7676578

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1587

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7656553, upper bound: 55.7176914
time: 84.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7176914, upper bound: 55.7656553
time: 354.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 438.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 438.59
Output dim: 13, lower bound: -55.7656553, upper bound: 55.7176914
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 438.59
Output dim: 13, lower bound: -55.7176914, upper bound: 55.7656553

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 783

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7646517, upper bound: 55.7048842
time: 65.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7048842, upper bound: 55.7166863
time: 73.93 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 673

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7150323, upper bound: 55.7629713
time: 78.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7149954, upper bound: 55.7630099
time: 61.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 141.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 141.76
Output dim: 13, lower bound: -55.7646517, upper bound: 55.7048842
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 141.76
Output dim: 13, lower bound: -55.7048842, upper bound: 55.7166863
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 141.76
Output dim: 13, lower bound: -55.7150323, upper bound: 55.7629713
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 141.76
Output dim: 13, lower bound: -55.7149954, upper bound: 55.7630099

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7623259, upper bound: 55.7033261
time: 83.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7630968, upper bound: 55.7025604
time: 86.82 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 641

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1590

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7481819, upper bound: 55.7153289
time: 64.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7514367, upper bound: 55.7120498
time: 74.30 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 691

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.7093262, upper bound: 55.7005074
time: 74.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6525974, upper bound: 55.7571849
time: 68.19 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1600

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 593

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7038386, upper bound: 55.7583915
time: 76.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7038386, upper bound: 55.7518654
time: 64.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 143.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7623259, upper bound: 55.7033261
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7630968, upper bound: 55.7025604
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7481819, upper bound: 55.7153289
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7514367, upper bound: 55.7120498
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7093262, upper bound: 55.7005074
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.6525974, upper bound: 55.7571849
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7038386, upper bound: 55.7583915
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 143.33
Output dim: 13, lower bound: -55.7038386, upper bound: 55.7518654

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7606519, upper bound: 55.6811312
time: 82.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7404767, upper bound: 55.7019728
time: 71.98 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 727

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7614304, upper bound: 55.6803617
time: 79.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7412515, upper bound: 55.7012053
time: 78.14 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7481668, upper bound: 55.7142783
time: 82.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7471271, upper bound: 55.7153138
time: 80.99 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7513066, upper bound: 55.7116473
time: 117.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7510330, upper bound: 55.7119186
time: 79.28 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 869

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6357210, upper bound: 55.7524978
time: 78.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6481338, upper bound: 55.7402080
time: 116.83 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 868

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6697092, upper bound: 55.7421712
time: 69.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.6876010, upper bound: 55.7242692
time: 102.05 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7087634, upper bound: 55.7515904
time: 71.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7035637, upper bound: 55.7502533
time: 68.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 142.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7606519, upper bound: 55.6811312
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7404767, upper bound: 55.7019728
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7614304, upper bound: 55.6803617
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7412515, upper bound: 55.7012053
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7481668, upper bound: 55.7142783
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7471271, upper bound: 55.7153138
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7513066, upper bound: 55.7116473
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7510330, upper bound: 55.7119186
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.6357210, upper bound: 55.7524978
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.6481338, upper bound: 55.7402080
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.6697092, upper bound: 55.7421712
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.6876010, upper bound: 55.7242692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7087634, upper bound: 55.7515904
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 142.60
Output dim: 13, lower bound: -55.7035637, upper bound: 55.7502533

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7579256, upper bound: 55.6444753
time: 66.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7189870, upper bound: 55.6783226
time: 73.63 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1712

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7330160, upper bound: 55.6927690
time: 74.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7320051, upper bound: 55.6943371
time: 81.31 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 886

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.6807545, upper bound: 55.5997827
time: 86.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.6807545, upper bound: 55.5997827
time: 88.59 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1579

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7398018, upper bound: 55.6983744
time: 80.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7384127, upper bound: 55.6997437
time: 67.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 839

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7175421, upper bound: 55.6760830
time: 66.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -55.7101065, upper bound: 55.6837206
time: 84.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1581

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7383682, upper bound: 55.7065232
time: 73.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7383682, upper bound: 55.7145062
time: 65.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=368, inp2_unstable=368, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1557

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7503577, upper bound: 55.7104856
time: 70.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -55.7501779, upper bound: 55.7106802
time: 78.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 150.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7579256, upper bound: 55.6444753
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7189870, upper bound: 55.6783226
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7330160, upper bound: 55.6927690
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7320051, upper bound: 55.6943371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.6807545, upper bound: 55.5997827
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.6807545, upper bound: 55.5997827
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7398018, upper bound: 55.6983744
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7384127, upper bound: 55.6997437
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7175421, upper bound: 55.6760830
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7101065, upper bound: 55.6837206
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7383682, upper bound: 55.7065232
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7383682, upper bound: 55.7145062
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7503577, upper bound: 55.7104856
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 150.54
Output dim: 13, lower bound: -55.7501779, upper bound: 55.7106802
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.7510330, upper bound: 55.7119186
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.6357210, upper bound: 55.7524978
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.6481338, upper bound: 55.7402080
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.6697092, upper bound: 55.7421712
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.6876010, upper bound: 55.7242692
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.7087634, upper bound: 55.7515904
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 150.54
Output dim: 13, lower bound: -55.7035637, upper bound: 55.7502533

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 89.00 + 3591.32 = 3680.32 seconds
