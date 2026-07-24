## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 7200 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686)
1: (-79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412)
2: (-74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370)
3: (-82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477)
4: (-86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703)
5: (-85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421)
6: (-119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241)
7: (-102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548)
8: (-107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637)
9: (-82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453)
10: (-123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886)
11: (-123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185)
12: (-120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276)
13: (-129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174)
14: (-189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811)
15: (-91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333)
16: (-128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881)
17: (-187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033)
18: (-124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420)
19: (-90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827)
20: (-85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634)
21: (-114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401)
22: (-120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228)
23: (-90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354)
24: (-115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648)
25: (-97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353)
26: (-134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830)
27: (-122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984)
28: (-89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180)
29: (-128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650)
30: (-115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049)
31: (-117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805)
32: (-122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201)
33: (-153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904)
34: (-127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389)
35: (-124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868)
36: (-128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347)
37: (-175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275)
38: (-154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509)
39: (-170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571)
40: (-142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118)
41: (-122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392)
42: (-90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.06 + 200.69 = 203.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -149.6355114, upper bound: 149.6355114

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5906116, upper bound: 149.5852100
time: 309.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5906116, upper bound: 149.5906116
time: 182.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 492.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 492.16
Output dim: 11, lower bound: -149.5906116, upper bound: 149.5852100
IS_A2, status: Status.UNKNOWN, split count: 1, time: 492.16
Output dim: 11, lower bound: -149.5906116, upper bound: 149.5906116

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -153.4958496, 90.6894455, -153.5822754, 90.7220306, -244.2178650, 244.2717285
1: -79.2239761, 71.0725861, -79.2565765, 71.0915985, -150.3155823, 150.3291626
2: -74.7687531, 74.6232605, -74.8482285, 74.6490936, -149.4178467, 149.4714966
3: -82.5805664, 88.0885849, -82.6567078, 88.1263199, -170.7068481, 170.7453003
4: -86.5005875, 86.4454651, -86.6035080, 86.4724197, -172.9729919, 173.0489807
5: -85.2149277, 89.3767700, -85.3049011, 89.4217682, -174.6366882, 174.6816711
6: -119.7718124, 91.5253830, -119.8158035, 91.6074219, -211.3792419, 211.3411713
7: -102.3201599, 82.1205368, -102.3898926, 82.1553802, -184.4755402, 184.5104370
8: -107.3038330, 106.9020844, -107.3989410, 106.9379044, -214.2417297, 214.3010254
9: -82.8174438, 88.1460114, -82.8623810, 88.1893921, -171.0068359, 171.0083923
10: -122.9241180, 114.6170425, -122.9745026, 114.6815796, -237.6056976, 237.5915375
11: -123.0660400, 70.4828339, -123.1299744, 70.5653381, -193.6313629, 193.6127930
12: -120.2699356, 118.8007355, -120.3186340, 119.0353622, -239.3052979, 239.1193542
13: -129.5948944, 133.3258667, -129.6361084, 133.4563293, -263.0512085, 262.9619446
14: -189.5117493, 119.2154617, -189.5733643, 119.4222336, -308.9339600, 308.7888184
15: -91.3851776, 83.6129913, -91.4691925, 83.6584778, -175.0436554, 175.0821838
16: -128.6837616, 85.9889526, -128.7495728, 86.0245819, -214.7083282, 214.7385254
17: -187.7438354, 120.7224045, -187.8038177, 120.9468689, -308.6907043, 308.5262146
18: -124.4546890, 104.6576233, -124.5100250, 104.7059479, -229.1606445, 229.1676483
19: -90.4959717, 45.7087097, -90.5461807, 45.7212982, -136.2172699, 136.2548828
20: -84.9442749, 61.0196877, -84.9842911, 61.0573158, -146.0015869, 146.0039825
21: -114.2711792, 57.6654587, -114.3225632, 57.6869011, -171.9580841, 171.9880219
22: -120.5441971, 68.9448395, -120.5964661, 68.9858398, -189.5300140, 189.5412598
23: -90.4192734, 65.3167267, -90.4603119, 65.3376770, -155.7569427, 155.7770386
24: -115.3683777, 67.8023148, -115.4432526, 67.8209000, -183.1892548, 183.2455750
25: -97.8101730, 70.5343246, -97.8591080, 70.5641785, -168.3743591, 168.3934326
26: -134.0042114, 110.7619171, -134.0597534, 110.8484344, -244.8526459, 244.8216400
27: -122.3532639, 86.1951447, -122.4309616, 86.2230682, -208.5763092, 208.6260986
28: -89.8415375, 73.9924927, -89.8807831, 74.0191727, -163.8607178, 163.8732758
29: -128.4714508, 65.9165192, -128.5139465, 65.9860687, -194.4575043, 194.4304657
30: -114.9660568, 89.7126389, -115.0105515, 89.7766113, -204.7426758, 204.7231903
31: -117.5920715, 62.1937561, -117.6568451, 62.2121887, -179.8042603, 179.8505859
32: -122.0090561, 88.3287582, -122.0480652, 88.4453735, -210.4544067, 210.3768311
33: -153.0140381, 106.6683502, -153.1428833, 106.7072220, -259.7212524, 259.8112183
34: -127.2981186, 88.4586487, -127.3448181, 88.4943924, -215.7925110, 215.8034515
35: -124.3865280, 86.3766708, -124.4478989, 86.4030075, -210.7895203, 210.8245697
36: -128.7757263, 96.0787659, -128.8133545, 96.1464691, -224.9221954, 224.8920898
37: -175.5478516, 93.6099472, -175.6388855, 93.6377945, -269.1856384, 269.2488098
38: -154.4918823, 118.7677994, -154.5442200, 118.8209534, -273.3128357, 273.3120117
39: -170.2432861, 110.9598770, -170.3287659, 110.9842224, -281.2275085, 281.2886047
40: -142.6728058, 94.5067749, -142.7519531, 94.5329742, -237.2057648, 237.2587280
41: -122.1452942, 91.9150085, -122.2024689, 91.9472351, -214.0925293, 214.1174622
42: -90.3281097, 80.4803543, -90.3674927, 80.5335464, -170.8616333, 170.8478394

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5827077
time: 174.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5827077
time: 190.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -153.8559265, 91.1313324, -153.6567078, 90.7542877, -244.6102142, 244.7880402
1: -79.4385910, 71.3770599, -79.2815170, 71.1097565, -150.5483398, 150.6585693
2: -75.0335236, 75.0486755, -74.9267807, 74.6736832, -149.7072144, 149.9754639
3: -82.8251190, 88.5652924, -82.7341309, 88.1603241, -170.9854431, 171.2994232
4: -86.8187943, 87.0413055, -86.7067947, 86.4957275, -173.3145142, 173.7481079
5: -85.4899139, 89.9447021, -85.3961639, 89.4633789, -174.9532928, 175.3408508
6: -120.2273712, 91.8121185, -119.8497314, 91.6757812, -211.9031525, 211.6618347
7: -102.6376572, 82.4202728, -102.4483643, 82.1880341, -184.8256836, 184.8686371
8: -107.6402283, 107.4275665, -107.4938965, 106.9705276, -214.6107483, 214.9214325
9: -83.1029663, 88.4049454, -82.9038773, 88.2232971, -171.3262634, 171.3088226
10: -123.5137100, 114.9421234, -123.0233154, 114.7319946, -238.2456970, 237.9654236
11: -123.8844681, 70.6663208, -123.1879959, 70.5857391, -194.4701996, 193.8543091
12: -121.2719803, 119.4372025, -120.3655472, 119.2814560, -240.5534210, 239.8027496
13: -130.0427246, 133.7823029, -129.6701660, 133.5867615, -263.6294861, 263.4524536
14: -190.3708954, 119.7259521, -189.6279297, 119.6446075, -310.0155029, 309.3538818
15: -91.6454315, 84.1072540, -91.4968719, 83.6986618, -175.3440857, 175.6041260
16: -129.2376709, 86.2355728, -128.8082886, 86.0530930, -215.2907715, 215.0438538
17: -188.8933716, 121.3007278, -187.8599091, 121.1780930, -310.0714722, 309.1606445
18: -124.8036957, 104.8871307, -124.5505981, 104.7494354, -229.5531311, 229.4377289
19: -90.9148178, 45.9066467, -90.5942688, 45.7313232, -136.6461334, 136.5009155
20: -85.2930908, 61.1697083, -85.0212708, 61.0902519, -146.3833313, 146.1909790
21: -114.8519669, 57.8202705, -114.3706665, 57.7063408, -172.5582886, 172.1909332
22: -120.9047546, 69.2331772, -120.6412659, 69.0233765, -189.9281311, 189.8744507
23: -90.7856979, 65.5266418, -90.4992981, 65.3558655, -156.1415710, 156.0259399
24: -115.6581421, 68.0684052, -115.5101700, 67.8367081, -183.4948425, 183.5785828
25: -98.0973587, 70.7984467, -97.8982849, 70.5890961, -168.6864471, 168.6967163
26: -134.6406860, 111.0670624, -134.1120758, 110.9302673, -245.5709534, 245.1791229
27: -122.6594543, 86.3944855, -122.4856339, 86.2457581, -208.9052124, 208.8800964
28: -90.1531982, 74.1359177, -89.9182129, 74.0439758, -164.1971741, 164.0541382
29: -128.9217224, 66.1635742, -128.5452271, 66.0601807, -194.9819031, 194.7088013
30: -115.2980042, 89.9425278, -115.0470734, 89.8074036, -205.1054077, 204.9895935
31: -118.0035324, 62.5250854, -117.7193146, 62.2266312, -180.2301636, 180.2444000
32: -122.4336472, 88.6596375, -122.0780716, 88.5620575, -210.9956970, 210.7377014
33: -153.4614716, 107.2040787, -153.2705536, 106.7392273, -260.2006836, 260.4746399
34: -127.5385895, 88.8736191, -127.3884811, 88.5242310, -216.0628204, 216.2621002
35: -124.6527328, 86.6659393, -124.5017700, 86.4235306, -211.0762634, 211.1677094
36: -129.1891022, 96.3449097, -128.8478699, 96.2182999, -225.4073486, 225.1927490
37: -176.0040588, 93.8485031, -175.7216492, 93.6625137, -269.6665649, 269.5701599
38: -154.9717255, 119.0351410, -154.5907593, 118.8712540, -273.8429871, 273.6259155
39: -170.6786499, 111.3346558, -170.4067841, 111.0001373, -281.6787720, 281.7414246
40: -143.0248566, 94.8417969, -142.8214111, 94.5524368, -237.5773010, 237.6631775
41: -122.5614090, 92.1179657, -122.2542496, 91.9750595, -214.5364685, 214.3722229
42: -90.8173828, 80.6997833, -90.4020386, 80.5703659, -171.3877563, 171.1018066

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5881087
time: 277.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5881086
time: 177.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 457.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 457.96
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5827077
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 457.96
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5827077
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 457.96
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5881087
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 457.96
Output dim: 11, lower bound: -149.5420516, upper bound: 149.5881086

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -153.2773132, 90.6251373, -153.1269379, 90.4342194, -243.7115173, 243.7520752
1: -79.1469803, 71.0286560, -79.0767822, 70.8995514, -150.0465393, 150.1054382
2: -74.5480957, 74.5691681, -74.4437637, 74.3755188, -148.9236145, 149.0129395
3: -82.3420563, 88.0148926, -82.2101135, 87.7912903, -170.1333313, 170.2250061
4: -86.2881012, 86.3711700, -86.2126160, 86.1549301, -172.4430237, 172.5837860
5: -85.0003052, 89.3020172, -84.9038391, 89.0592270, -174.0595093, 174.2058563
6: -119.6767578, 91.3475189, -119.4902115, 91.2371368, -210.9138947, 210.8377075
7: -102.1446991, 82.0498810, -102.0261536, 81.9131927, -184.0578766, 184.0760193
8: -107.1163101, 106.8370056, -107.0336227, 106.6927414, -213.8090515, 213.8706055
9: -82.6386795, 88.0599976, -82.5003281, 87.8655243, -170.5041809, 170.5603333
10: -122.7522736, 114.4982758, -122.5603104, 114.2827301, -237.0350037, 237.0585785
11: -122.9540024, 70.2597809, -122.5845795, 70.1577911, -193.1117859, 192.8443451
12: -120.1685181, 118.3770218, -119.7941971, 118.2621384, -238.4306641, 238.1712189
13: -129.4304199, 133.1935730, -129.3005524, 133.0876312, -262.5180664, 262.4941406
14: -189.3421326, 118.8173141, -188.9457550, 118.7132416, -308.0553589, 307.7630615
15: -91.1759186, 83.5328751, -91.0527191, 83.3801575, -174.5560760, 174.5855865
16: -128.5559692, 85.9131622, -128.4125671, 85.7865295, -214.3424988, 214.3257141
17: -187.5941162, 120.1967773, -186.9708862, 120.0136414, -307.6076965, 307.1676636
18: -124.3465195, 104.3460159, -124.0731201, 104.1412354, -228.4877625, 228.4191284
19: -90.3985901, 45.6637459, -90.1581955, 45.6125488, -136.0111389, 135.8219299
20: -84.8568573, 60.9320755, -84.6731873, 60.8697929, -145.7266388, 145.6052551
21: -114.1621552, 57.5634956, -113.8306656, 57.4755249, -171.6376801, 171.3941650
22: -120.4363403, 68.7925568, -120.1619263, 68.6738968, -189.1101990, 188.9544830
23: -90.3338013, 65.2782440, -90.1409454, 65.2435913, -155.5773773, 155.4191895
24: -115.2623444, 67.7291794, -115.1196365, 67.6605835, -182.9229279, 182.8488159
25: -97.7098312, 70.4707794, -97.5766144, 70.4111176, -168.1209412, 168.0473938
26: -133.8927765, 110.4175644, -133.4766235, 110.2279434, -244.1207275, 243.8941956
27: -122.2271423, 85.9650726, -121.9372787, 85.7994690, -208.0266113, 207.9023438
28: -89.7522278, 73.8477020, -89.5065384, 73.7433472, -163.4955750, 163.3542328
29: -128.3666992, 65.6938171, -127.9905701, 65.5807495, -193.9474182, 193.6843719
30: -114.8722610, 89.5002289, -114.6337509, 89.3747101, -204.2469788, 204.1339722
31: -117.4713898, 62.1418457, -117.2711334, 62.0739746, -179.5453644, 179.4129791
32: -121.9112167, 88.1295853, -121.7762909, 88.0495148, -209.9607239, 209.9058838
33: -152.7150269, 106.5779419, -152.5726624, 106.2366257, -258.9516602, 259.1506042
34: -127.1832581, 88.3858414, -127.0759277, 88.2180786, -215.4013062, 215.4617615
35: -124.2372208, 86.3234787, -124.1303024, 86.1888962, -210.4261017, 210.4537811
36: -128.6881409, 95.9125671, -128.4947205, 95.7952347, -224.4833679, 224.4072876
37: -175.3786316, 93.5245667, -175.2176208, 93.4263458, -268.8049927, 268.7421570
38: -154.3719940, 118.6489944, -154.1896667, 118.5050964, -272.8770752, 272.8386536
39: -170.0462646, 110.9019928, -169.9290924, 110.6068802, -280.6531372, 280.8310547
40: -142.5415649, 94.4507217, -142.4499512, 94.3361740, -236.8777313, 236.9006653
41: -122.0277405, 91.7936020, -121.9097137, 91.6718674, -213.6995850, 213.7033081
42: -90.2228394, 80.3891754, -90.0867004, 80.3125916, -170.5354309, 170.4758606

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5329207
time: 278.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5623211
time: 236.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -153.4776611, 90.6826935, -153.5509338, 90.7103119, -244.1879730, 244.2336273
1: -79.2164154, 71.0669098, -79.2434235, 71.0816498, -150.2980652, 150.3103180
2: -74.7557983, 74.6175079, -74.8257751, 74.6389923, -149.3947906, 149.4432831
3: -82.5668640, 88.0815735, -82.6328735, 88.1141205, -170.6809845, 170.7144470
4: -86.4880371, 86.4386826, -86.5817871, 86.4606628, -172.9486694, 173.0204773
5: -85.1979675, 89.3693390, -85.2752304, 89.4088745, -174.6068268, 174.6445618
6: -119.7637405, 91.4899826, -119.8017044, 91.5495911, -211.3133240, 211.2916870
7: -102.3065796, 82.1120758, -102.3661575, 82.1405945, -184.4471741, 184.4782410
8: -107.2899017, 106.8939362, -107.3746796, 106.9236450, -214.2135468, 214.2685852
9: -82.8060303, 88.1388397, -82.8425064, 88.1769104, -170.9829407, 170.9813232
10: -122.9039001, 114.6074677, -122.9388733, 114.6652832, -237.5691833, 237.5463104
11: -123.0570984, 70.4579849, -123.1144867, 70.5259247, -193.5830231, 193.5724640
12: -120.2604446, 118.7761307, -120.3021164, 118.9928207, -239.2532501, 239.0782166
13: -129.5550842, 133.3147736, -129.5660095, 133.4371490, -262.9921875, 262.8807373
14: -189.4964905, 119.1945953, -189.5469360, 119.3856964, -308.8821716, 308.7415161
15: -91.3550720, 83.6059113, -91.4171600, 83.6461792, -175.0012512, 175.0230408
16: -128.6722412, 85.9796143, -128.7295837, 86.0083084, -214.6805420, 214.7091827
17: -187.7300415, 120.6950073, -187.7798157, 120.8991699, -308.6291809, 308.4748230
18: -124.4440918, 104.6277084, -124.4915390, 104.6542435, -229.0983276, 229.1192474
19: -90.4878387, 45.7028198, -90.5321274, 45.7108841, -136.1987305, 136.2349548
20: -84.9371033, 61.0044365, -84.9718628, 61.0304642, -145.9675598, 145.9762878
21: -114.2624817, 57.6512985, -114.3074722, 57.6624107, -171.9248810, 171.9587708
22: -120.5318069, 68.9353485, -120.5748825, 68.9693298, -189.5011292, 189.5102234
23: -90.4120941, 65.3115997, -90.4478989, 65.3286591, -155.7407532, 155.7594910
24: -115.3563766, 67.7886658, -115.4224625, 67.7970581, -183.1534424, 183.2111053
25: -97.7983246, 70.5284500, -97.8384247, 70.5539398, -168.3522339, 168.3668823
26: -133.9926300, 110.7422256, -134.0397644, 110.8141479, -244.8067780, 244.7819824
27: -122.3427734, 86.1731110, -122.4128418, 86.1875458, -208.5303192, 208.5859375
28: -89.8347702, 73.9827423, -89.8691406, 74.0043030, -163.8390808, 163.8518829
29: -128.4589996, 65.9032288, -128.4920959, 65.9627533, -194.4217377, 194.3953247
30: -114.9564972, 89.6900940, -114.9938812, 89.7379074, -204.6943970, 204.6839752
31: -117.5821228, 62.1892548, -117.6396332, 62.2043610, -179.7864685, 179.8288879
32: -122.0003967, 88.3022995, -122.0330048, 88.3992310, -210.3996277, 210.3352966
33: -152.9921570, 106.6617203, -153.1050262, 106.6957245, -259.6878357, 259.7667542
34: -127.2869720, 88.4525757, -127.3255005, 88.4838104, -215.7707825, 215.7780762
35: -124.3639679, 86.3718109, -124.4091187, 86.3945312, -210.7584991, 210.7809296
36: -128.7685394, 96.0666046, -128.8009949, 96.1250687, -224.8936157, 224.8675995
37: -175.5323181, 93.5962296, -175.6118469, 93.6136551, -269.1459656, 269.2080688
38: -154.4814606, 118.7508698, -154.5262146, 118.7910690, -273.2725220, 273.2770691
39: -170.2251740, 110.9548035, -170.2978363, 110.9754028, -281.2005615, 281.2526245
40: -142.6612244, 94.4992065, -142.7318420, 94.5197906, -237.1810150, 237.2310486
41: -122.1367035, 91.8907776, -122.1875916, 91.9048615, -214.0415649, 214.0783691
42: -90.3211670, 80.4699402, -90.3555145, 80.5156860, -170.8368225, 170.8254547

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5329207
time: 148.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5623211
time: 158.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -153.6376801, 91.0676956, -153.2011719, 90.4667358, -244.1044159, 244.2688599
1: -79.3619232, 71.3334656, -79.1015625, 70.9178314, -150.2797546, 150.4350128
2: -74.8132935, 74.9949341, -74.5222244, 74.4003296, -149.2136078, 149.5171509
3: -82.5873871, 88.4921875, -82.2875366, 87.8256226, -170.4129944, 170.7797241
4: -86.6069794, 86.9674911, -86.3158035, 86.1785583, -172.7854919, 173.2832947
5: -85.2758865, 89.8703461, -84.9951706, 89.1012039, -174.3770905, 174.8655090
6: -120.1333313, 91.6336136, -119.5245056, 91.3049393, -211.4382629, 211.1581116
7: -102.4626465, 82.3506775, -102.0844498, 81.9462738, -184.4089050, 184.4351196
8: -107.4531708, 107.3628082, -107.1286316, 106.7256317, -214.1787720, 214.4914246
9: -82.9264145, 88.3183517, -82.5412292, 87.8995972, -170.8260040, 170.8595581
10: -123.3442078, 114.8223419, -122.6083603, 114.3333359, -237.6775208, 237.4306641
11: -123.7750015, 70.4442444, -122.6432800, 70.1784210, -193.9534149, 193.0874939
12: -121.1713562, 119.0145111, -119.8414993, 118.5083389, -239.6796875, 238.8560181
13: -129.8786163, 133.6488495, -129.3345947, 133.2181549, -263.0967712, 262.9834595
14: -190.2019348, 119.3277512, -189.0007935, 118.9354248, -309.1373291, 308.3285217
15: -91.4371262, 84.0273056, -91.0807953, 83.4206009, -174.8577271, 175.1080933
16: -129.1106110, 86.1596375, -128.4716797, 85.8154449, -214.9260559, 214.6313019
17: -188.7442932, 120.7752151, -187.0274963, 120.2444916, -308.9887695, 307.8027039
18: -124.6945801, 104.5759888, -124.1132355, 104.1845093, -228.8790894, 228.6892242
19: -90.8191910, 45.8620911, -90.2067337, 45.6224976, -136.4416809, 136.0688171
20: -85.2063293, 61.0820198, -84.7101822, 60.9026871, -146.1090088, 145.7922058
21: -114.7446976, 57.7189407, -113.8793030, 57.4948959, -172.2395630, 171.5982361
22: -120.7968292, 69.0815125, -120.2070694, 68.7111206, -189.5079498, 189.2885742
23: -90.7012939, 65.4883118, -90.1802826, 65.2617340, -155.9630280, 155.6685944
24: -115.5516281, 67.9956970, -115.1869278, 67.6762924, -183.2279205, 183.1826172
25: -97.9968872, 70.7354507, -97.6160507, 70.4358749, -168.4327698, 168.3515015
26: -134.5299835, 110.7234421, -133.5294495, 110.3096924, -244.8396606, 244.2528687
27: -122.5322266, 86.1660080, -121.9921646, 85.8215179, -208.3537445, 208.1581726
28: -90.0641937, 73.9918976, -89.5443420, 73.7679367, -163.8321228, 163.5362396
29: -128.8173828, 65.9409027, -128.0220184, 65.6546783, -194.4720612, 193.9629211
30: -115.2044525, 89.7304535, -114.6707153, 89.4056015, -204.6100464, 204.4011536
31: -117.8829651, 62.4735107, -117.3342514, 62.0884781, -179.9714203, 179.8077698
32: -122.3364410, 88.4608002, -121.8064041, 88.1660004, -210.5024414, 210.2672119
33: -153.1626282, 107.1142349, -152.7001343, 106.2689285, -259.4315491, 259.8143616
34: -127.4244843, 88.8007202, -127.1196289, 88.2480774, -215.6725311, 215.9203186
35: -124.5035095, 86.6131439, -124.1840973, 86.2097244, -210.7132263, 210.7972412
36: -129.1018982, 96.1785278, -128.5294495, 95.8668060, -224.9686890, 224.7079468
37: -175.8356628, 93.7630463, -175.3006744, 93.4510117, -269.2866821, 269.0637207
38: -154.8525085, 118.9156113, -154.2364197, 118.5550842, -273.4075317, 273.1520386
39: -170.4818115, 111.2771988, -170.0068359, 110.6229248, -281.1047363, 281.2840271
40: -142.8933258, 94.7855377, -142.5194092, 94.3557129, -237.2490387, 237.3049469
41: -122.4451141, 91.9967346, -121.9618149, 91.6996689, -214.1447754, 213.9585266
42: -90.7128601, 80.6082001, -90.1213531, 80.3492584, -171.0621033, 170.7295532

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1185

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5384114
time: 157.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5676989
time: 166.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -153.8377838, 91.1246490, -153.6254425, 90.7425766, -244.5803528, 244.7500763
1: -79.4310989, 71.3713989, -79.2683640, 71.0998230, -150.5309143, 150.6397552
2: -75.0205612, 75.0429382, -74.9043274, 74.6636047, -149.6841431, 149.9472656
3: -82.8114014, 88.5583344, -82.7103043, 88.1481323, -170.9595337, 171.2686157
4: -86.8062286, 87.0345612, -86.6850586, 86.4839554, -173.2901764, 173.7196198
5: -85.4729919, 89.9373016, -85.3664780, 89.4504929, -174.9234772, 175.3037567
6: -120.2193832, 91.7764587, -119.8356323, 91.6181793, -211.8375549, 211.6120758
7: -102.6241150, 82.4118347, -102.4246216, 82.1732254, -184.7973328, 184.8364410
8: -107.6263123, 107.4194260, -107.4696426, 106.9562531, -214.5825500, 214.8890381
9: -83.0916443, 88.3977432, -82.8840027, 88.2107849, -171.3024292, 171.2817383
10: -123.4936523, 114.9325256, -122.9874268, 114.7156906, -238.2093201, 237.9199371
11: -123.8757324, 70.6415253, -123.1725311, 70.5465393, -194.4222565, 193.8140259
12: -121.2625427, 119.4125748, -120.3490295, 119.2389908, -240.5014954, 239.7615967
13: -130.0029907, 133.7710876, -129.6000366, 133.5673828, -263.5703430, 263.3711243
14: -190.3556519, 119.7050705, -189.6014404, 119.6080627, -309.9637146, 309.3065186
15: -91.6153870, 84.1002502, -91.4447937, 83.6863861, -175.3017578, 175.5450134
16: -129.2262573, 86.2262268, -128.7883301, 86.0368271, -215.2630615, 215.0145569
17: -188.8796539, 121.2732544, -187.8359222, 121.1303482, -310.0099487, 309.1091919
18: -124.7929459, 104.8572922, -124.5319519, 104.6978149, -229.4907532, 229.3892517
19: -90.9067841, 45.9008102, -90.5802155, 45.7207947, -136.6275787, 136.4810181
20: -85.2860031, 61.1544647, -85.0088806, 61.0633430, -146.3493347, 146.1633453
21: -114.8433380, 57.8061562, -114.3555832, 57.6817932, -172.5251312, 172.1617279
22: -120.8923798, 69.2237854, -120.6196747, 69.0068588, -189.8992310, 189.8434601
23: -90.7785492, 65.5215759, -90.4868927, 65.3467712, -156.1253204, 156.0084534
24: -115.6461105, 68.0548019, -115.4893723, 67.8127899, -183.4588928, 183.5441742
25: -98.0855255, 70.7926636, -97.8776093, 70.5788116, -168.6643219, 168.6702576
26: -134.6292114, 111.0474014, -134.0921021, 110.8959961, -245.5252075, 245.1394806
27: -122.6489258, 86.3721924, -122.4675293, 86.2103348, -208.8592224, 208.8397217
28: -90.1464539, 74.1260986, -89.9065857, 74.0293732, -164.1758270, 164.0326691
29: -128.9093018, 66.1502380, -128.5233612, 66.0368423, -194.9461365, 194.6735840
30: -115.2884750, 89.9199982, -115.0304260, 89.7686920, -205.0571594, 204.9504089
31: -117.9935684, 62.5205994, -117.7021332, 62.2188110, -180.2123718, 180.2227325
32: -122.4250488, 88.6332092, -122.0630188, 88.5159760, -210.9410248, 210.6962280
33: -153.4395905, 107.1975021, -153.2326050, 106.7277069, -260.1672974, 260.4301147
34: -127.5275192, 88.8675690, -127.3691559, 88.5136261, -216.0411377, 216.2367249
35: -124.6302032, 86.6611023, -124.4629898, 86.4150848, -211.0452423, 211.1240845
36: -129.1819305, 96.3327560, -128.8354797, 96.1968689, -225.3787689, 225.1682281
37: -175.9885559, 93.8348236, -175.6945496, 93.6383362, -269.6268921, 269.5293579
38: -154.9613342, 119.0182800, -154.5727692, 118.8412704, -273.8026123, 273.5910645
39: -170.6605682, 111.3296051, -170.3758698, 110.9913177, -281.6518860, 281.7054749
40: -143.0133057, 94.8342209, -142.8011780, 94.5392761, -237.5525818, 237.6354065
41: -122.5529022, 92.0938263, -122.2393875, 91.9326859, -214.4855957, 214.3332214
42: -90.8105011, 80.6893845, -90.3900528, 80.5525360, -171.3630066, 171.0794373

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1185

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5384114
time: 4140.99 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5676989
time: 226.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4369.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5329207
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5623211
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5329207
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5623211
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5384114
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5676989
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5384114
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4369.60
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5676989

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -152.7725830, 90.2983856, -152.9231415, 90.3428116, -243.1153870, 243.2214966
1: -78.9140091, 70.7722626, -78.9936218, 70.8457489, -149.7597656, 149.7658691
2: -74.0971375, 74.2146530, -74.2230225, 74.3205109, -148.4176331, 148.4376678
3: -82.0124817, 87.6282578, -82.0626221, 87.6853638, -169.6978455, 169.6908569
4: -85.7551346, 85.9077148, -85.9492874, 86.0749741, -171.8301086, 171.8569946
5: -84.5907288, 88.8956375, -84.7139282, 88.9561462, -173.5468750, 173.6095581
6: -119.2638168, 90.9637451, -119.3524551, 91.0639038, -210.3277130, 210.3161926
7: -101.6459045, 81.7257233, -101.8141327, 81.8270035, -183.4729004, 183.5398560
8: -106.6027069, 106.3637085, -106.7868652, 106.5963364, -213.1990204, 213.1505585
9: -82.2520905, 87.6410828, -82.4200897, 87.6701431, -169.9222412, 170.0611725
10: -121.8680954, 113.6793442, -122.4353180, 113.8485870, -235.7166748, 236.1146240
11: -122.2264404, 69.7905579, -122.4500580, 69.9107590, -192.1371918, 192.2406158
12: -119.3047562, 117.2310867, -119.6895218, 117.6442566, -236.9490051, 236.9206085
13: -129.0563354, 132.5507812, -129.2043762, 132.8012085, -261.8575439, 261.7551575
14: -188.4866943, 117.8909607, -188.7857361, 118.1919098, -306.6785889, 306.6766968
15: -90.7558670, 83.1200256, -90.8768997, 83.2643814, -174.0202332, 173.9969177
16: -127.9482346, 85.6753693, -128.2637634, 85.7005005, -213.6487427, 213.9391327
17: -186.8031006, 119.2321167, -186.8565369, 119.5093842, -306.3124695, 306.0886536
18: -123.8361206, 103.9689026, -123.9209366, 103.9724884, -227.8086090, 227.8898315
19: -89.9647675, 45.3765869, -90.0013809, 45.5756149, -135.5403748, 135.3779602
20: -84.4994354, 60.7443771, -84.5675201, 60.7827911, -145.2822266, 145.3118896
21: -113.6906128, 57.4031982, -113.6978455, 57.4300423, -171.1206512, 171.1010284
22: -119.9525452, 68.4596710, -119.9662323, 68.5679016, -188.5204468, 188.4259033
23: -90.0378036, 65.0803070, -90.0424957, 65.1931076, -155.2309113, 155.1228027
24: -114.7009277, 67.4731750, -114.8510361, 67.6200562, -182.3209686, 182.3241882
25: -97.3674622, 70.1935730, -97.4380646, 70.3339996, -167.7014618, 167.6316376
26: -133.2569275, 109.8462524, -133.3258972, 109.9519501, -243.2088776, 243.1721191
27: -121.5296021, 85.6386642, -121.5964127, 85.7445526, -207.2741547, 207.2350769
28: -89.4031830, 73.5574493, -89.3549271, 73.6840897, -163.0872803, 162.9123688
29: -127.9589233, 65.3952484, -127.8352051, 65.4545364, -193.4134369, 193.2304535
30: -114.5508118, 89.1944733, -114.5152435, 89.2503967, -203.8012085, 203.7097168
31: -116.8811188, 61.8505707, -117.0435715, 62.0298996, -178.9109955, 178.8941345
32: -121.4706879, 87.6385803, -121.6543884, 87.7882996, -209.2589722, 209.2929688
33: -152.0319824, 106.0627899, -152.2210693, 106.1520157, -258.1839905, 258.2838440
34: -126.9064178, 88.0189819, -126.9594727, 88.1317444, -215.0381622, 214.9784546
35: -123.7249908, 85.8755035, -123.8710327, 86.1192703, -209.8442688, 209.7465363
36: -128.3620148, 95.6961288, -128.3673706, 95.7285309, -224.0905457, 224.0635071
37: -174.7344971, 93.1679688, -174.9360046, 93.3595963, -268.0940857, 268.1039429
38: -153.8824005, 118.3130569, -154.0317383, 118.3866959, -272.2690735, 272.3447876
39: -169.4538574, 110.4627686, -169.6763916, 110.5509796, -280.0048218, 280.1391602
40: -141.9830170, 94.1948090, -142.2194214, 94.2879944, -236.2710114, 236.4142151
41: -121.6427612, 91.5181274, -121.7606201, 91.6050797, -213.2478333, 213.2787476
42: -89.7383118, 79.9038239, -89.9933624, 80.0889511, -169.8272705, 169.8971710

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1185

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.4600208, upper bound: 149.5256398
time: 172.11 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -149.4600208, upper bound: 149.5256398
time: 363.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 538.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 538.41
Output dim: 11, lower bound: -149.4600208, upper bound: 149.5256398
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 538.41
Output dim: 11, lower bound: -149.4600208, upper bound: 149.5256398
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5623211
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5329207
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5623211
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5384114
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5220433, upper bound: 149.5676989
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5384114
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 538.41
Output dim: 11, lower bound: -149.5676988, upper bound: 149.5676989

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 203.75 + 7378.41 = 7582.17 seconds
