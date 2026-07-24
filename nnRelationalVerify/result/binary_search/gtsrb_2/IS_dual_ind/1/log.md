## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 151.123838933
Search space: {k/256.0 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 2.81 + 176.04 = 178.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -156.8589948, upper bound: 156.8589950


# Binary Search by BASE starts (time budget: 17821.15 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=193.89141845703125
rel_dist={11: [-153.0152404825352, 153.01524066488776]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=193.89141845703125
rel_dist={11: [-149.63551136954993, 149.6355114190527]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=193.89141845703125
rel_dist={11: [-150.9499420800349, 150.94994213534193]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=193.89141845703125
rel_dist={11: [-152.0594572987635, 152.05945742807847]}

## Binary Search Result
Binary search time: 831.54 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.015625


# Individual Split (IS_dual_ind) starts
Time budget: 16989.60 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

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
Output dim: 11, lower bound: -154.5060549, upper bound: 154.4959056
time: 169.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4974050, upper bound: 154.4974052
time: 1404.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1573.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1573.57
Output dim: 11, lower bound: -154.5060549, upper bound: 154.4959056
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1573.57
Output dim: 11, lower bound: -154.4974050, upper bound: 154.4974052

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -153.4958496, 90.6894455, -153.6791382, 90.7581711, -244.2539673, 244.3685913
1: -79.2239761, 71.0725861, -79.2927856, 71.1127930, -150.3367615, 150.3653564
2: -74.7687531, 74.6232605, -74.9357758, 74.6776123, -149.4463501, 149.5590363
3: -82.5805664, 88.0885849, -82.7404480, 88.1683197, -170.7488708, 170.8290253
4: -86.5005875, 86.4454651, -86.7170868, 86.5026245, -173.0031891, 173.1625519
5: -85.2149277, 89.3767700, -85.4034882, 89.4710388, -174.6859436, 174.7802582
6: -119.7718124, 91.5253830, -119.8654633, 91.6995697, -211.4713745, 211.3908386
7: -102.3201599, 82.1205368, -102.4677277, 82.1937256, -184.5138855, 184.5882568
8: -107.3038330, 106.9020844, -107.5039062, 106.9776306, -214.2814331, 214.4059448
9: -82.8174438, 88.1460114, -82.9121552, 88.2379303, -171.0553741, 171.0581512
10: -122.9241180, 114.6170425, -123.0309601, 114.7529449, -237.6770477, 237.6479950
11: -123.0660400, 70.4828339, -123.2013397, 70.6586914, -193.7247009, 193.6841736
12: -120.2699356, 118.8007355, -120.3719482, 119.2927017, -239.5626221, 239.1726837
13: -129.5948944, 133.3258667, -129.6817932, 133.6056976, -263.2005920, 263.0076294
14: -189.5117493, 119.2154617, -189.6413879, 119.6485519, -309.1602783, 308.8568420
15: -91.3851776, 83.6129913, -91.5622482, 83.7093506, -175.0945282, 175.1752319
16: -128.6837616, 85.9889526, -128.8228149, 86.0645447, -214.7483063, 214.8117676
17: -187.7438354, 120.7224045, -187.8699188, 121.1951828, -308.9390259, 308.5923157
18: -124.4546890, 104.6576233, -124.5739441, 104.7601852, -229.2148438, 229.2315674
19: -90.4959717, 45.7087097, -90.6022797, 45.7353439, -136.2313232, 136.3109894
20: -84.9442749, 61.0196877, -85.0290833, 61.0987816, -146.0430450, 146.0487671
21: -114.2711792, 57.6654587, -114.3800049, 57.7106400, -171.9818115, 172.0454712
22: -120.5441971, 68.9448395, -120.6547775, 69.0322418, -189.5764313, 189.5996094
23: -90.4192734, 65.3167267, -90.5058289, 65.3610382, -155.7803040, 155.8225403
24: -115.3683777, 67.8023148, -115.5264435, 67.8414154, -183.2097778, 183.3287354
25: -97.8101730, 70.5343246, -97.9134445, 70.5972748, -168.4074402, 168.4477539
26: -134.0042114, 110.7619171, -134.1214600, 110.9447403, -244.9489441, 244.8833618
27: -122.3532639, 86.1951447, -122.5189896, 86.2540588, -208.6073303, 208.7141113
28: -89.8415375, 73.9924927, -89.9240417, 74.0486984, -163.8902283, 163.9165344
29: -128.4714508, 65.9165192, -128.5614929, 66.0641937, -194.5356445, 194.4780121
30: -114.9660568, 89.7126389, -115.0605850, 89.8475647, -204.8136292, 204.7732239
31: -117.5920715, 62.1937561, -117.7294540, 62.2326698, -179.8247375, 179.9232178
32: -122.0090561, 88.3287582, -122.0917282, 88.5737152, -210.5827332, 210.4204712
33: -153.0140381, 106.6683502, -153.2844543, 106.7505951, -259.7645874, 259.9528198
34: -127.2981186, 88.4586487, -127.3964310, 88.5346603, -215.8327789, 215.8550720
35: -124.3865280, 86.3766708, -124.5156631, 86.4327698, -210.8193054, 210.8923340
36: -128.7757263, 96.0787659, -128.8548889, 96.2220078, -224.9977417, 224.9336548
37: -175.5478516, 93.6099472, -175.7394409, 93.6687241, -269.2165833, 269.3493958
38: -154.4918823, 118.7677994, -154.6018677, 118.8815384, -273.3734131, 273.3696594
39: -170.2432861, 110.9598770, -170.4235535, 111.0116425, -281.2549438, 281.3834229
40: -142.6728058, 94.5067749, -142.8411713, 94.5619278, -237.2346802, 237.3479462
41: -122.1452942, 91.9150085, -122.2659683, 91.9827423, -214.1280212, 214.1809692
42: -90.3281097, 80.4803543, -90.4116364, 80.5935974, -170.9216919, 170.8919983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
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
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 602
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
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1630
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
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1716
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1791
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1217
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 887
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
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 654
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
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1125
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
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1616
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
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 972
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1603
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
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1778
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

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4557567, upper bound: 154.4750671
time: 344.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4557567, upper bound: 154.4728851
time: 242.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -153.8559265, 91.1313324, -153.6796265, 90.7602997, -244.6162262, 244.8109589
1: -79.4385910, 71.3770599, -79.2914429, 71.1137238, -150.5523071, 150.6685028
2: -75.0335236, 75.0486755, -74.9411774, 74.6789703, -149.7124939, 149.9898376
3: -82.8251190, 88.5652924, -82.7466278, 88.1692505, -170.9943695, 171.3119202
4: -86.8187943, 87.0413055, -86.7245255, 86.5027313, -173.3215332, 173.7658386
5: -85.4899139, 89.9447021, -85.4108887, 89.4730072, -174.9629059, 175.3555756
6: -120.2273712, 91.8121185, -119.8636322, 91.6980286, -211.9253998, 211.6757507
7: -102.6376572, 82.4202728, -102.4672699, 82.1954041, -184.8330688, 184.8875122
8: -107.6402283, 107.4275665, -107.5105057, 106.9788361, -214.6190338, 214.9380341
9: -83.1029663, 88.4049454, -82.9138260, 88.2367401, -171.3397064, 171.3187714
10: -123.5137100, 114.9421234, -123.0334167, 114.7515259, -238.2652283, 237.9755402
11: -123.8844681, 70.6663208, -123.2028198, 70.6377106, -194.5221558, 193.8691406
12: -121.2719803, 119.4372025, -120.3749237, 119.3151855, -240.5871582, 239.8121338
13: -130.0427246, 133.7823029, -129.6816101, 133.6124573, -263.6551819, 263.4639282
14: -190.3708954, 119.7259521, -189.6427612, 119.6714096, -310.0422974, 309.3687134
15: -91.6454315, 84.1072540, -91.5434036, 83.7098236, -175.3552551, 175.6506653
16: -129.2376709, 86.2355728, -128.8242035, 86.0638275, -215.3014984, 215.0597839
17: -188.8933716, 121.3007278, -187.8726807, 121.2142029, -310.1075745, 309.1734009
18: -124.8036957, 104.8871307, -124.5704727, 104.7612076, -229.5649109, 229.4575958
19: -90.9148178, 45.9066467, -90.6046524, 45.7350845, -136.6499023, 136.5112915
20: -85.2930908, 61.1697083, -85.0303040, 61.0997162, -146.3928070, 146.2000122
21: -114.8519669, 57.8202705, -114.3818207, 57.7113266, -172.5632935, 172.2020874
22: -120.9047546, 69.2331772, -120.6549072, 69.0327454, -189.9374847, 189.8880920
23: -90.7856979, 65.5266418, -90.5077438, 65.3612823, -156.1469727, 156.0343781
24: -115.6581421, 68.0684052, -115.5281601, 67.8415680, -183.4996948, 183.5965576
25: -98.0973587, 70.7984467, -97.9126892, 70.5971298, -168.6944733, 168.7111206
26: -134.6406860, 111.0670624, -134.1237793, 110.9487305, -245.5894165, 245.1908264
27: -122.6594543, 86.3944855, -122.5136871, 86.2538147, -208.9132690, 208.9081726
28: -90.1531982, 74.1359177, -89.9261017, 74.0497742, -164.2029724, 164.0620117
29: -128.9217224, 66.1635742, -128.5595093, 66.0703354, -194.9920654, 194.7230835
30: -115.2980042, 89.9425278, -115.0598373, 89.8384323, -205.1364441, 205.0023499
31: -118.0035324, 62.5250854, -117.7324677, 62.2321968, -180.2357330, 180.2575073
32: -122.4336472, 88.6596375, -122.0904083, 88.5821838, -211.0158386, 210.7500458
33: -153.4614716, 107.2040787, -153.2937317, 106.7500610, -260.2115479, 260.4978027
34: -127.5385895, 88.8736191, -127.3984451, 88.5343399, -216.0729065, 216.2720642
35: -124.6527328, 86.6659393, -124.5168304, 86.4316559, -211.0843811, 211.1827698
36: -129.1891022, 96.3449097, -128.8562775, 96.2279129, -225.4170227, 225.2011719
37: -176.0040588, 93.8485031, -175.7424622, 93.6692810, -269.6733398, 269.5909424
38: -154.9717255, 119.0351410, -154.6031799, 118.8833847, -273.8551025, 273.6383057
39: -170.6786499, 111.3346558, -170.4264069, 111.0092316, -281.6878662, 281.7610474
40: -143.0248566, 94.8417969, -142.8417969, 94.5608826, -237.5857086, 237.6835785
41: -122.5614090, 92.1179657, -122.2674561, 91.9829788, -214.5443878, 214.3854218
42: -90.8173828, 80.6997833, -90.4119263, 80.5899734, -171.4073486, 171.1116943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1718
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
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
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
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 602
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1590
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
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1630
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
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1584
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1217
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
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 658
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
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1125
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
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 635
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
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 756
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4765410
time: 283.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4743085
time: 182.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 468.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 468.90
Output dim: 11, lower bound: -154.4557567, upper bound: 154.4750671
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 468.90
Output dim: 11, lower bound: -154.4557567, upper bound: 154.4728851
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 468.90
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4765410
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 468.90
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4743085

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -153.4044189, 90.6628952, -153.2230377, 90.4704514, -243.8748779, 243.8859253
1: -79.1921082, 71.0544815, -79.1123047, 70.9207001, -150.1128082, 150.1667786
2: -74.6782532, 74.6010132, -74.5311432, 74.4040985, -149.0823517, 149.1321411
3: -82.4813385, 88.0582581, -82.2936096, 87.8333588, -170.3146973, 170.3518677
4: -86.4132690, 86.4148560, -86.3260040, 86.1851044, -172.5983734, 172.7408447
5: -85.1268616, 89.3460007, -85.0023575, 89.1087418, -174.2355957, 174.3483582
6: -119.7325974, 91.4522247, -119.5396271, 91.3287811, -211.0613708, 210.9918518
7: -102.2480087, 82.0913544, -102.1027832, 81.9518127, -184.1998138, 184.1941223
8: -107.2268372, 106.8752747, -107.1383209, 106.7324829, -213.9593201, 214.0135803
9: -82.7431412, 88.1106873, -82.5493469, 87.9139252, -170.6570740, 170.6600342
10: -122.8527298, 114.5682907, -122.6154861, 114.3542023, -237.2069397, 237.1837769
11: -123.0198212, 70.3902893, -122.6558838, 70.2509232, -193.2707214, 193.0461731
12: -120.2282104, 118.6270905, -119.8477249, 118.5191650, -238.7473755, 238.4748230
13: -129.5271606, 133.2715454, -129.3460083, 133.2355347, -262.7626953, 262.6175537
14: -189.4420319, 119.0524597, -189.0139465, 118.9390411, -308.3810425, 308.0664062
15: -91.2979279, 83.5799484, -91.1452789, 83.4307098, -174.7286377, 174.7251892
16: -128.6310272, 85.9577484, -128.4858398, 85.8264465, -214.4574585, 214.4435883
17: -187.6822815, 120.5067215, -187.0372925, 120.2611847, -307.9434509, 307.5440063
18: -124.4103241, 104.5253906, -124.1352158, 104.1952209, -228.6055450, 228.6606140
19: -90.4559021, 45.6901360, -90.2144470, 45.6264496, -136.0823364, 135.9045715
20: -84.9082031, 60.9836426, -84.7176208, 60.9110947, -145.8192902, 145.7012482
21: -114.2263641, 57.6235886, -113.8881378, 57.4990959, -171.7254639, 171.5117188
22: -120.4998245, 68.8820648, -120.2202682, 68.7190399, -189.2188568, 189.1022949
23: -90.3841248, 65.3008575, -90.1864700, 65.2667999, -155.6509247, 155.4873352
24: -115.3247833, 67.7721100, -115.2027130, 67.6809235, -183.0057068, 182.9748230
25: -97.7688293, 70.5080032, -97.6309662, 70.4437408, -168.2125702, 168.1389618
26: -133.9583435, 110.6207809, -133.5384521, 110.3234482, -244.2817383, 244.1592255
27: -122.3014679, 86.0980530, -122.0247726, 85.8297424, -208.1312103, 208.1228027
28: -89.8048172, 73.9325867, -89.5500412, 73.7725143, -163.5773163, 163.4826355
29: -128.4283447, 65.8250732, -128.0379791, 65.6577911, -194.0861359, 193.8630524
30: -114.9274063, 89.6230164, -114.6836395, 89.4453430, -204.3727264, 204.3066559
31: -117.5423431, 62.1723442, -117.3436279, 62.0943413, -179.6366882, 179.5159607
32: -121.9686279, 88.2466888, -121.8198090, 88.1775284, -210.1461487, 210.0664978
33: -152.8914490, 106.6310730, -152.7138367, 106.2798004, -259.1712341, 259.3449097
34: -127.2507706, 88.4285583, -127.1272736, 88.2581711, -215.5089417, 215.5558167
35: -124.3251190, 86.3546829, -124.1977921, 86.2184601, -210.5435791, 210.5524750
36: -128.7397461, 96.0105057, -128.5363159, 95.8695068, -224.6092377, 224.5468140
37: -175.4782867, 93.5746613, -175.3180695, 93.4570770, -268.9353638, 268.8927307
38: -154.4425354, 118.7187805, -154.2474213, 118.5650406, -273.0075684, 272.9661865
39: -170.1620483, 110.9359741, -170.0233612, 110.6338196, -280.7958374, 280.9593506
40: -142.6188202, 94.4837799, -142.5381775, 94.3649368, -236.9837341, 237.0219269
41: -122.0969086, 91.8650513, -121.9730530, 91.7071381, -213.8040466, 213.8380585
42: -90.2846603, 80.4428253, -90.1304626, 80.3722992, -170.6569519, 170.5732880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 648
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
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 645
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
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1720
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
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 589
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
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
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
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 660
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
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
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
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 736
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
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1125
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
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1169
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
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1648
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
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1723
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
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1681
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
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 968
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
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4258344, upper bound: 154.4252037
time: 1097.52 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.4158561, upper bound: 154.4353779
time: 1685.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2785.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2785.78
Output dim: 11, lower bound: -154.4258344, upper bound: 154.4252037
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2785.78
Output dim: 11, lower bound: -154.4158561, upper bound: 154.4353779
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2785.78
Output dim: 11, lower bound: -154.4557567, upper bound: 154.4728851
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2785.78
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4765410
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2785.78
Output dim: 11, lower bound: -154.4470258, upper bound: 154.4743085
Binary search (step 0): status=Status.UNKNOWN, k_low=5, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=193.89141845703125
rel_dist={11: [-154.55252351539193, 154.55252373789625]}

## Binary search (step 1) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9633475, upper bound: 152.9565716
time: 283.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9593232, upper bound: 152.9593233
time: 183.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 467.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 467.28
Output dim: 11, lower bound: -152.9633475, upper bound: 152.9565716
IS_A2, status: Status.UNKNOWN, split count: 1, time: 467.28
Output dim: 11, lower bound: -152.9593232, upper bound: 152.9593233

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -153.4958496, 90.6894455, -153.6454468, 90.7457199, -244.2415771, 244.3348999
1: -79.2239761, 71.0725861, -79.2801208, 71.1054382, -150.3294067, 150.3527069
2: -74.7687531, 74.6232605, -74.9055634, 74.6677399, -149.4364929, 149.5288086
3: -82.5805664, 88.0885849, -82.7115631, 88.1537476, -170.7343140, 170.8001404
4: -86.5005875, 86.4454651, -86.6779175, 86.4920731, -172.9926453, 173.1233673
5: -85.2149277, 89.3767700, -85.3695450, 89.4540710, -174.6690063, 174.7463074
6: -119.7718124, 91.5253830, -119.8480988, 91.6676331, -211.4394531, 211.3734741
7: -102.3201599, 82.1205368, -102.4406052, 82.1804504, -184.5006104, 184.5611420
8: -107.3038330, 106.9020844, -107.4676514, 106.9638367, -214.2676544, 214.3697357
9: -82.8174438, 88.1460114, -82.8949203, 88.2210236, -171.0384674, 171.0409241
10: -122.9241180, 114.6170425, -123.0112457, 114.7283020, -237.6524200, 237.6282959
11: -123.0660400, 70.4828339, -123.1765442, 70.6264114, -193.6924133, 193.6593475
12: -120.2699356, 118.8007355, -120.3535995, 119.2041702, -239.4740906, 239.1543121
13: -129.5948944, 133.3258667, -129.6659851, 133.5527344, -263.1476440, 262.9918518
14: -189.5117493, 119.2154617, -189.6178284, 119.5707626, -309.0825195, 308.8332825
15: -91.3851776, 83.6129913, -91.5299759, 83.6916275, -175.0768127, 175.1429749
16: -128.6837616, 85.9889526, -128.7973938, 86.0505981, -214.7343597, 214.7863464
17: -187.7438354, 120.7224045, -187.8471527, 121.1095886, -308.8534241, 308.5695496
18: -124.4546890, 104.6576233, -124.5514984, 104.7413788, -229.1960754, 229.2091064
19: -90.4959717, 45.7087097, -90.5828552, 45.7304611, -136.2264252, 136.2915649
20: -84.9442749, 61.0196877, -85.0135574, 61.0844498, -146.0287170, 146.0332489
21: -114.2711792, 57.6654587, -114.3600540, 57.7024269, -171.9736023, 172.0255127
22: -120.5441971, 68.9448395, -120.6344147, 69.0159454, -189.5601196, 189.5792236
23: -90.4192734, 65.3167267, -90.4900513, 65.3529510, -155.7722168, 155.8067780
24: -115.3683777, 67.8023148, -115.4975281, 67.8343201, -183.2026672, 183.2998352
25: -97.8101730, 70.5343246, -97.8945923, 70.5858002, -168.3959656, 168.4288940
26: -134.0042114, 110.7619171, -134.1001282, 110.9112930, -244.9154968, 244.8620453
27: -122.3532639, 86.1951447, -122.4881210, 86.2433014, -208.5965576, 208.6832581
28: -89.8415375, 73.9924927, -89.9091034, 74.0383759, -163.8799133, 163.9015961
29: -128.4714508, 65.9165192, -128.5448608, 66.0370255, -194.5084686, 194.4613800
30: -114.9660568, 89.7126389, -115.0431137, 89.8230057, -204.7890625, 204.7557526
31: -117.5920715, 62.1937561, -117.7041473, 62.2255592, -179.8176270, 179.8978882
32: -122.0090561, 88.3287582, -122.0765381, 88.5294266, -210.5384521, 210.4052887
33: -153.0140381, 106.6683502, -153.2357330, 106.7355423, -259.7495422, 259.9040833
34: -127.2981186, 88.4586487, -127.3785858, 88.5206757, -215.8187866, 215.8372192
35: -124.3865280, 86.3766708, -124.4922791, 86.4223328, -210.8088684, 210.8689423
36: -128.7757263, 96.0787659, -128.8405457, 96.1956177, -224.9713440, 224.9192963
37: -175.5478516, 93.6099472, -175.7046509, 93.6580200, -269.2058716, 269.3146057
38: -154.4918823, 118.7677994, -154.5819702, 118.8604736, -273.3523560, 273.3497620
39: -170.2432861, 110.9598770, -170.3907471, 111.0019913, -281.2452698, 281.3506165
40: -142.6728058, 94.5067749, -142.8100586, 94.5518570, -237.2246552, 237.3168335
41: -122.1452942, 91.9150085, -122.2439423, 91.9703674, -214.1156311, 214.1589355
42: -90.3281097, 80.4803543, -90.3962555, 80.5726929, -170.9007874, 170.8766022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1719
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
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
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
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1585
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
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 752
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
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 755
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1704
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
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1540
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
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 733
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
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 569
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
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1079
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
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1039
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
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 1081
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
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9116586, upper bound: 152.9457963
time: 188.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9116586, upper bound: 152.9457363
time: 173.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -153.8559265, 91.1313324, -153.6721191, 90.7583313, -244.6142578, 244.8034515
1: -79.4385910, 71.3770599, -79.2881775, 71.1124191, -150.5509949, 150.6652374
2: -75.0335236, 75.0486755, -74.9364471, 74.6772308, -149.7107544, 149.9851227
3: -82.8251190, 88.5652924, -82.7425690, 88.1662979, -170.9914246, 171.3078613
4: -86.8187943, 87.0413055, -86.7186737, 86.5004272, -173.3192139, 173.7599792
5: -85.4899139, 89.9447021, -85.4061050, 89.4698105, -174.9597168, 175.3507996
6: -120.2273712, 91.8121185, -119.8590240, 91.6906738, -211.9180450, 211.6711426
7: -102.6376572, 82.4202728, -102.4610977, 82.1929855, -184.8306427, 184.8813477
8: -107.6402283, 107.4275665, -107.5050812, 106.9760895, -214.6163177, 214.9326172
9: -83.1029663, 88.4049454, -82.9105225, 88.2322769, -171.3352356, 171.3154602
10: -123.5137100, 114.9421234, -123.0301514, 114.7451019, -238.2588043, 237.9722595
11: -123.8844681, 70.6663208, -123.1979828, 70.6208954, -194.5053406, 193.8643036
12: -121.2719803, 119.4372025, -120.3718338, 119.3041382, -240.5760803, 239.8090363
13: -130.0427246, 133.7823029, -129.6778412, 133.6035309, -263.6462402, 263.4601440
14: -190.3708954, 119.7259521, -189.6379089, 119.6625900, -310.0334473, 309.3638611
15: -91.6454315, 84.1072540, -91.5282516, 83.7061462, -175.3515778, 175.6354980
16: -129.2376709, 86.2355728, -128.8189392, 86.0603104, -215.2979736, 215.0545044
17: -188.8933716, 121.3007278, -187.8684998, 121.2021790, -310.0955505, 309.1691895
18: -124.8036957, 104.8871307, -124.5638885, 104.7573395, -229.5610352, 229.4510193
19: -90.9148178, 45.9066467, -90.6011963, 45.7338562, -136.6486664, 136.5078430
20: -85.2930908, 61.1697083, -85.0273361, 61.0966339, -146.3897247, 146.1970520
21: -114.8519669, 57.8202705, -114.3781204, 57.7096863, -172.5616455, 172.1983948
22: -120.9047546, 69.2331772, -120.6504211, 69.0296783, -189.9344177, 189.8835907
23: -90.7856979, 65.5266418, -90.5049667, 65.3594971, -156.1452026, 156.0316162
24: -115.6581421, 68.0684052, -115.5222397, 67.8399734, -183.4981079, 183.5906372
25: -98.0973587, 70.7984467, -97.9079742, 70.5944672, -168.6918030, 168.7064209
26: -134.6406860, 111.0670624, -134.1198730, 110.9427032, -245.5833893, 245.1869202
27: -122.6594543, 86.3944855, -122.5041885, 86.2511749, -208.9106293, 208.8986816
28: -90.1531982, 74.1359177, -89.9235077, 74.0478668, -164.2010651, 164.0594177
29: -128.9217224, 66.1635742, -128.5548401, 66.0670013, -194.9887238, 194.7184143
30: -115.2980042, 89.9425278, -115.0556717, 89.8283386, -205.1263428, 204.9981842
31: -118.0035324, 62.5250854, -117.7281036, 62.2303619, -180.2338867, 180.2531738
32: -122.4336472, 88.6596375, -122.0863495, 88.5755844, -211.0092316, 210.7459564
33: -153.4614716, 107.2040787, -153.2861023, 106.7465515, -260.2080078, 260.4901733
34: -127.5385895, 88.8736191, -127.3951721, 88.5309906, -216.0695801, 216.2687683
35: -124.6527328, 86.6659393, -124.5118484, 86.4289932, -211.0817261, 211.1777802
36: -129.1891022, 96.3449097, -128.8534851, 96.2247925, -225.4138794, 225.1983643
37: -176.0040588, 93.8485031, -175.7355652, 93.6670837, -269.6711426, 269.5840454
38: -154.9717255, 119.0351410, -154.5990906, 118.8793564, -273.8510742, 273.6342163
39: -170.6786499, 111.3346558, -170.4199371, 111.0062637, -281.6849060, 281.7545776
40: -143.0248566, 94.8417969, -142.8350220, 94.5581360, -237.5829773, 237.6768188
41: -122.5614090, 92.1179657, -122.2630844, 91.9804077, -214.5417786, 214.3810425
42: -90.8173828, 80.6997833, -90.4086838, 80.5835419, -171.4009247, 171.1084595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
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
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 602
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1590
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
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 634
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
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1584
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
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1125
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
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 635
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
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9076691, upper bound: 152.9482627
time: 164.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9076691, upper bound: 152.9481671
time: 1043.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 1210.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1210.31
Output dim: 11, lower bound: -152.9116586, upper bound: 152.9457963
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1210.31
Output dim: 11, lower bound: -152.9116586, upper bound: 152.9457363
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1210.31
Output dim: 11, lower bound: -152.9076691, upper bound: 152.9482627
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1210.31
Output dim: 11, lower bound: -152.9076691, upper bound: 152.9481671

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -153.3628845, 90.6505585, -153.1898346, 90.4580154, -243.8208923, 243.8403931
1: -79.1774750, 71.0461121, -79.1000671, 70.9133987, -150.0908508, 150.1461792
2: -74.6361084, 74.5907059, -74.5010376, 74.3942871, -149.0303955, 149.0917358
3: -82.4362793, 88.0442352, -82.2649307, 87.8188629, -170.2551422, 170.3091431
4: -86.3727646, 86.4006958, -86.2869415, 86.1746902, -172.5474396, 172.6876373
5: -85.0858765, 89.3317642, -84.9684906, 89.0917358, -174.1776123, 174.3002472
6: -119.7144318, 91.4182968, -119.5225525, 91.2970734, -211.0115051, 210.9408569
7: -102.2144470, 82.0778732, -102.0764236, 81.9384995, -184.1529541, 184.1542969
8: -107.1909790, 106.8628922, -107.1022644, 106.7187729, -213.9097595, 213.9651184
9: -82.7090225, 88.0942383, -82.5324554, 87.8972015, -170.6062164, 170.6266937
10: -122.8201218, 114.5456009, -122.5963821, 114.3295822, -237.1497040, 237.1419830
11: -122.9983444, 70.3480530, -122.6313553, 70.2187653, -193.2171021, 192.9794006
12: -120.2088470, 118.5461807, -119.8293686, 118.4309158, -238.6397400, 238.3755493
13: -129.4958344, 133.2462769, -129.3303833, 133.1835938, -262.6794128, 262.5766602
14: -189.4096985, 118.9765854, -188.9904175, 118.8615341, -308.2712402, 307.9670105
15: -91.2579041, 83.5646515, -91.1133041, 83.4132919, -174.6712036, 174.6779480
16: -128.6066589, 85.9432678, -128.4605713, 85.8125610, -214.4192047, 214.4038391
17: -187.6537781, 120.4064026, -187.0144806, 120.1760101, -307.8297424, 307.4208984
18: -124.3897018, 104.4657135, -124.1137924, 104.1765976, -228.5662994, 228.5794678
19: -90.4373169, 45.6815643, -90.1950684, 45.6216393, -136.0589294, 135.8766327
20: -84.8914642, 60.9669380, -84.7022934, 60.8968620, -145.7883301, 145.6692352
21: -114.2055435, 57.6041565, -113.8683319, 57.4909821, -171.6965027, 171.4724884
22: -120.4792404, 68.8529968, -120.2000580, 68.7034607, -189.1827087, 189.0530548
23: -90.3678131, 65.2935028, -90.1707916, 65.2588043, -155.6266174, 155.4642944
24: -115.3045959, 67.7581635, -115.1739807, 67.6739349, -182.9785004, 182.9321289
25: -97.7497330, 70.4958191, -97.6121597, 70.4325256, -168.1822510, 168.1079712
26: -133.9371033, 110.5551834, -133.5172119, 110.2905197, -244.2276306, 244.0723724
27: -122.2773514, 86.0550842, -121.9943848, 85.8192673, -208.0966187, 208.0494690
28: -89.7877655, 73.9050598, -89.5350952, 73.7623749, -163.5501404, 163.4401550
29: -128.4083862, 65.7826385, -128.0214996, 65.6313095, -194.0397034, 193.8041229
30: -114.9094925, 89.5823822, -114.6664124, 89.4209366, -204.3304291, 204.2487946
31: -117.5193176, 62.1624374, -117.3186264, 62.0873299, -179.6066284, 179.4810486
32: -121.9499969, 88.2088013, -121.8047409, 88.1333923, -210.0833893, 210.0135193
33: -152.8344421, 106.6137848, -152.6652985, 106.2649612, -259.0993958, 259.2790527
34: -127.2288666, 88.4146729, -127.1096191, 88.2443085, -215.4731750, 215.5242767
35: -124.2966461, 86.3445129, -124.1745529, 86.2082672, -210.5048676, 210.5190735
36: -128.7230377, 95.9788361, -128.5220032, 95.8438721, -224.5669098, 224.5008240
37: -175.4460449, 93.5583649, -175.2834930, 93.4464951, -268.8925476, 268.8418579
38: -154.4196625, 118.6961288, -154.2275696, 118.5443192, -272.9639893, 272.9237061
39: -170.1245117, 110.9248505, -169.9908752, 110.6245270, -280.7490234, 280.9157104
40: -142.5937347, 94.4730759, -142.5076752, 94.3549194, -236.9486542, 236.9807434
41: -122.0744553, 91.8418808, -121.9512482, 91.6949005, -213.7693481, 213.7931061
42: -90.2645874, 80.4254456, -90.1153717, 80.3515930, -170.6161652, 170.5408020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 645
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
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1720
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 589
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
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
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
type: A, layer: 1, pos: 663
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
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 533
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 736
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
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1155
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
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1125
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
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1169
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
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
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
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1723
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
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1647
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
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 968
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
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8797200, upper bound: 152.8947734
time: 175.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8778599, upper bound: 152.9118770
time: 381.97 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -153.4874725, 90.6863708, -153.6139221, 90.7339630, -244.2214203, 244.3002930
1: -79.2205124, 71.0700150, -79.2667847, 71.0954285, -150.3159332, 150.3367920
2: -74.7627869, 74.6206360, -74.8830566, 74.6576157, -149.4204102, 149.5036926
3: -82.5743103, 88.0853806, -82.6876297, 88.1414795, -170.7157898, 170.7730103
4: -86.4947357, 86.4423676, -86.6561508, 86.4802017, -172.9749451, 173.0985107
5: -85.2072144, 89.3733521, -85.3397827, 89.4411316, -174.6483459, 174.7131348
6: -119.7681122, 91.5090332, -119.8338318, 91.6098938, -211.3780060, 211.3428650
7: -102.3139801, 82.1166534, -102.4165115, 82.1655960, -184.4795837, 184.5331573
8: -107.2974243, 106.8983765, -107.4432907, 106.9494781, -214.2469025, 214.3416443
9: -82.8121796, 88.1427383, -82.8750076, 88.2084274, -171.0205994, 171.0177307
10: -122.9149017, 114.6125793, -122.9752350, 114.7119598, -237.6268616, 237.5878143
11: -123.0619507, 70.4714966, -123.1608200, 70.5870514, -193.6490021, 193.6323090
12: -120.2656250, 118.7893982, -120.3370438, 119.1614914, -239.4271240, 239.1264343
13: -129.5769653, 133.3207550, -129.5957947, 133.5328674, -263.1098328, 262.9165344
14: -189.5048065, 119.2059174, -189.5912781, 119.5341034, -309.0389099, 308.7971802
15: -91.3708344, 83.6097565, -91.4777832, 83.6791229, -175.0499420, 175.0875244
16: -128.6784821, 85.9846878, -128.7772827, 86.0342484, -214.7127380, 214.7619629
17: -187.7375641, 120.7097321, -187.8230896, 121.0617065, -308.7992554, 308.5328369
18: -124.4498444, 104.6440430, -124.5325012, 104.6896515, -229.1394958, 229.1765137
19: -90.4922180, 45.7060242, -90.5686951, 45.7199554, -136.2121735, 136.2747192
20: -84.9409637, 61.0127945, -85.0010300, 61.0575409, -145.9985046, 146.0138245
21: -114.2671661, 57.6588402, -114.3448029, 57.6778870, -171.9450531, 172.0036316
22: -120.5385132, 68.9405060, -120.6127625, 68.9990997, -189.5376129, 189.5532532
23: -90.4159927, 65.3143845, -90.4775162, 65.3438416, -155.7598267, 155.7918854
24: -115.3628769, 67.7961426, -115.4765930, 67.8104172, -183.1732941, 183.2727356
25: -97.8047791, 70.5316315, -97.8738480, 70.5754395, -168.3801880, 168.4054871
26: -133.9989014, 110.7529526, -134.0800018, 110.8767548, -244.8756409, 244.8329315
27: -122.3484039, 86.1843872, -122.4697723, 86.2077942, -208.5561981, 208.6541595
28: -89.8383942, 73.9876404, -89.8974152, 74.0235977, -163.8619995, 163.8850555
29: -128.4658051, 65.9104614, -128.5229492, 66.0134125, -194.4792175, 194.4334106
30: -114.9617233, 89.7023468, -115.0262604, 89.7842484, -204.7459717, 204.7286072
31: -117.5874786, 62.1916771, -117.6866913, 62.2176819, -179.8051605, 179.8783722
32: -122.0050812, 88.3167114, -122.0614014, 88.4832687, -210.4883423, 210.3780975
33: -153.0039978, 106.6653214, -153.1977386, 106.7238617, -259.7278442, 259.8630676
34: -127.2929993, 88.4558716, -127.3591690, 88.5099487, -215.8029480, 215.8150330
35: -124.3761978, 86.3744583, -124.4534149, 86.4137268, -210.7899170, 210.8278503
36: -128.7723999, 96.0732117, -128.8281097, 96.1738281, -224.9462280, 224.9013062
37: -175.5407410, 93.6037598, -175.6774292, 93.6338196, -269.1745605, 269.2811890
38: -154.4870453, 118.7601395, -154.5639343, 118.8304214, -273.3174744, 273.3240662
39: -170.2349091, 110.9575577, -170.3598022, 110.9929962, -281.2279053, 281.3173523
40: -142.6674500, 94.5033112, -142.7896118, 94.5386276, -237.2060852, 237.2929077
41: -122.1413345, 91.9038773, -122.2289276, 91.9279633, -214.0693054, 214.1327972
42: -90.3248901, 80.4755402, -90.3841324, 80.5547714, -170.8796692, 170.8596802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 855
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
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1549
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
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 758
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
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8797200, upper bound: 152.8947252
time: 157.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8778599, upper bound: 152.9118048
time: 192.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -153.7232361, 91.0928802, -153.2166138, 90.4708023, -244.1940308, 244.3094940
1: -79.3923416, 71.3507690, -79.1082458, 70.9205017, -150.3128357, 150.4589996
2: -74.9011383, 75.0163498, -74.5319366, 74.4038925, -149.3050232, 149.5482788
3: -82.6814270, 88.5212708, -82.2960052, 87.8315887, -170.5130157, 170.8172760
4: -86.6914215, 86.9968033, -86.3276825, 86.1832962, -172.8747253, 173.3244934
5: -85.3611755, 89.8999100, -85.0051498, 89.1076660, -174.4688416, 174.9050598
6: -120.1705627, 91.7046661, -119.5338593, 91.3197479, -211.4903107, 211.2385254
7: -102.5321579, 82.3782959, -102.0972137, 81.9512634, -184.4834137, 184.4755096
8: -107.5276337, 107.3885269, -107.1398163, 106.7312164, -214.2588501, 214.5283356
9: -82.9956512, 88.3527374, -82.5478516, 87.9085388, -170.9041901, 170.9005890
10: -123.4111862, 114.8701019, -122.6151810, 114.3464355, -237.7576294, 237.4852753
11: -123.8182678, 70.5321655, -122.6533203, 70.2135315, -194.0317993, 193.1854858
12: -121.2114029, 119.1832352, -119.8478241, 118.5310364, -239.7424316, 239.0310364
13: -129.9438629, 133.7019501, -129.3422852, 133.2351379, -263.1790161, 263.0442200
14: -190.2691803, 119.4870300, -189.0108337, 118.9534073, -309.2225647, 308.4978638
15: -91.5187378, 84.0589752, -91.1118774, 83.4281464, -174.9468689, 175.1708527
16: -129.1609344, 86.1898041, -128.4824219, 85.8225708, -214.9835052, 214.6722107
17: -188.8037262, 120.9847488, -187.0360870, 120.2686310, -309.0723572, 308.0208130
18: -124.7379990, 104.6953888, -124.1263351, 104.1924820, -228.9304810, 228.8217163
19: -90.8572006, 45.8797379, -90.2137299, 45.6250267, -136.4822235, 136.0934601
20: -85.2406769, 61.1169128, -84.7162628, 60.9090424, -146.1497192, 145.8331757
21: -114.7873535, 57.7593498, -113.8868179, 57.4982224, -172.2855682, 171.6461639
22: -120.8397675, 69.1417007, -120.2162323, 68.7174606, -189.5572205, 189.3579407
23: -90.7348633, 65.5034943, -90.1860046, 65.2653732, -156.0002441, 155.6894989
24: -115.5940399, 68.0245056, -115.1989670, 67.6795807, -183.2736206, 183.2234802
25: -98.0368271, 70.7602692, -97.6257019, 70.4412308, -168.4780426, 168.3859406
26: -134.5740051, 110.8607178, -133.5372925, 110.3221588, -244.8961334, 244.3980103
27: -122.5828552, 86.2555618, -122.0107498, 85.8269348, -208.4097748, 208.2662964
28: -90.0995636, 74.0490112, -89.5496979, 73.7718353, -163.8713989, 163.5986938
29: -128.8588867, 66.0297089, -128.0316467, 65.6615067, -194.5203857, 194.0613403
30: -115.2415314, 89.8124542, -114.6793137, 89.4264221, -204.6679382, 204.4917603
31: -117.9308548, 62.4939499, -117.3430939, 62.0922165, -180.0230713, 179.8370361
32: -122.3749847, 88.5399933, -121.8147125, 88.1794891, -210.5544434, 210.3547058
33: -153.2820129, 107.1497879, -152.7156677, 106.2762299, -259.5582275, 259.8654480
34: -127.4698334, 88.8296051, -127.1263199, 88.2548599, -215.7247009, 215.9559326
35: -124.5629272, 86.6340256, -124.1941757, 86.2151718, -210.7780914, 210.8282013
36: -129.1365967, 96.2448578, -128.5350647, 95.8732758, -225.0098724, 224.7799225
37: -175.9027100, 93.7968445, -175.3146057, 93.4555817, -269.3582764, 269.1114502
38: -154.8999329, 118.9629593, -154.2448120, 118.5632019, -273.4631042, 273.2077637
39: -170.5599670, 111.2998734, -170.0199738, 110.6290970, -281.1890564, 281.3198547
40: -142.9455872, 94.8079529, -142.5329590, 94.3613281, -237.3069000, 237.3408813
41: -122.4912262, 92.0449524, -121.9706802, 91.7049561, -214.1961517, 214.0156250
42: -90.7543030, 80.6446228, -90.1280212, 80.3624191, -171.1166992, 170.7726440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1717
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
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1613
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
type: A, layer: 1, pos: 663
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
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1643
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
type: A, layer: 1, pos: 887
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
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
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
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 688
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
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1102
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
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1749
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1601
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
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1082
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
Output dim: 11, lower bound: -152.8756211, upper bound: 152.8974719
time: 153.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8736221, upper bound: 152.9139741
time: 426.07 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -153.8475647, 91.1282806, -153.6408691, 90.7466202, -244.5941772, 244.7691498
1: -79.4351425, 71.3744965, -79.2750397, 71.1024780, -150.5376129, 150.6495361
2: -75.0275574, 75.0460663, -74.9140015, 74.6671600, -149.6947174, 149.9600677
3: -82.8188477, 88.5621033, -82.7187500, 88.1540985, -170.9729462, 171.2808533
4: -86.8129654, 87.0382385, -86.6969376, 86.4887085, -173.3016663, 173.7351685
5: -85.4821930, 89.9412918, -85.3764038, 89.4569092, -174.9391022, 175.3176880
6: -120.2236938, 91.7957611, -119.8449554, 91.6330872, -211.8567810, 211.6407166
7: -102.6314697, 82.4164047, -102.4373779, 82.1781921, -184.8096619, 184.8537598
8: -107.6338196, 107.4238358, -107.4807892, 106.9618378, -214.5956421, 214.9046326
9: -83.0977554, 88.4016571, -82.8906631, 88.2197876, -171.3175049, 171.2923126
10: -123.5045700, 114.9376373, -122.9942703, 114.7287827, -238.2333527, 237.9318848
11: -123.8804703, 70.6550369, -123.1825256, 70.5816956, -194.4621582, 193.8375549
12: -121.2676697, 119.4258194, -120.3553543, 119.2616196, -240.5292664, 239.7811584
13: -130.0247955, 133.7771454, -129.6077271, 133.5842133, -263.6090088, 263.3848267
14: -190.3639374, 119.7164078, -189.6114349, 119.6260376, -309.9899597, 309.3278503
15: -91.6311340, 84.1040649, -91.4761505, 83.6938782, -175.3249817, 175.5802155
16: -129.2324524, 86.2313080, -128.7990112, 86.0440369, -215.2764740, 215.0303192
17: -188.8871155, 121.2879868, -187.8445129, 121.1544876, -310.0415955, 309.1324768
18: -124.7987518, 104.8735733, -124.5452271, 104.7056885, -229.5044403, 229.4187927
19: -90.9111023, 45.9039917, -90.5871506, 45.7233353, -136.6344299, 136.4911346
20: -85.2898254, 61.1628151, -85.0149307, 61.0697021, -146.3595276, 146.1777496
21: -114.8479767, 57.8136711, -114.3630600, 57.6851387, -172.5331116, 172.1767273
22: -120.8990784, 69.2288971, -120.6287994, 69.0131836, -189.9122620, 189.8576660
23: -90.7824249, 65.5243378, -90.4925766, 65.3503952, -156.1328125, 156.0169067
24: -115.6526184, 68.0622635, -115.5014343, 67.8160782, -183.4686890, 183.5636902
25: -98.0919571, 70.7957840, -97.8872910, 70.5841827, -168.6761475, 168.6830750
26: -134.6354065, 111.0580826, -134.0999146, 110.9084473, -245.5438385, 245.1579895
27: -122.6545715, 86.3835754, -122.4860535, 86.2157593, -208.8703308, 208.8696289
28: -90.1500854, 74.1310883, -89.9118729, 74.0332794, -164.1833649, 164.0429688
29: -128.9160919, 66.1574936, -128.5330048, 66.0436935, -194.9597473, 194.6904907
30: -115.2936707, 89.9322281, -115.0390167, 89.7896423, -205.0833130, 204.9712524
31: -117.9989319, 62.5230179, -117.7109222, 62.2225456, -180.2214661, 180.2339478
32: -122.4297180, 88.6475830, -122.0712662, 88.5295029, -210.9592285, 210.7188416
33: -153.4514160, 107.2010727, -153.2481689, 106.7350006, -260.1864014, 260.4492188
34: -127.5335083, 88.8708115, -127.3758392, 88.5204163, -216.0539093, 216.2466431
35: -124.6424103, 86.6637421, -124.4730682, 86.4205322, -211.0629272, 211.1368103
36: -129.1857758, 96.3393555, -128.8410950, 96.2033691, -225.3891296, 225.1804504
37: -175.9969177, 93.8423233, -175.7084656, 93.6428986, -269.6398010, 269.5507812
38: -154.9669189, 119.0274811, -154.5811005, 118.8493729, -273.8162842, 273.6085815
39: -170.6703033, 111.3323746, -170.3890076, 110.9974213, -281.6677246, 281.7213745
40: -143.0195312, 94.8383408, -142.8148193, 94.5449448, -237.5644836, 237.6531677
41: -122.5574875, 92.1069183, -122.2482147, 91.9379883, -214.4954834, 214.3551178
42: -90.8142014, 80.6949768, -90.3967285, 80.5656967, -171.3798981, 171.0916748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1621
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
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 749
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
type: A, layer: 1, pos: 887
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
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1155
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 987
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8756211, upper bound: 152.8973858
time: 454.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.8736221, upper bound: 152.9138831
time: 351.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 808.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8797200, upper bound: 152.8947734
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8778599, upper bound: 152.9118770
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8797200, upper bound: 152.8947252
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8778599, upper bound: 152.9118048
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8756211, upper bound: 152.8974719
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8736221, upper bound: 152.9139741
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8756211, upper bound: 152.8973858
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 808.73
Output dim: 11, lower bound: -152.8736221, upper bound: 152.9138831
Binary search (step 1): status=Status.UNKNOWN, k_low=5, k_high=7, k_mid=6, eps_mid=0.0234375, abs_max=193.89141845703125
rel_dist={11: [-153.0152404825352, 153.01524066488776]}

## Binary search (step 2) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.0086350, upper bound: 152.0030612
time: 151.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.0059169, upper bound: 152.0059169
time: 204.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 356.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 356.67
Output dim: 11, lower bound: -152.0086350, upper bound: 152.0030612
IS_A2, status: Status.UNKNOWN, split count: 1, time: 356.67
Output dim: 11, lower bound: -152.0059169, upper bound: 152.0059169

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -153.4958496, 90.6894455, -153.6262817, 90.7385635, -244.2344055, 244.3157349
1: -79.2239761, 71.0725861, -79.2729111, 71.1012192, -150.3251953, 150.3454895
2: -74.7687531, 74.6232605, -74.8882904, 74.6621094, -149.4308319, 149.5115356
3: -82.5805664, 88.0885849, -82.6950073, 88.1454239, -170.7259827, 170.7835999
4: -86.5005875, 86.4454651, -86.6554718, 86.4860764, -172.9866638, 173.1009369
5: -85.2149277, 89.3767700, -85.3500824, 89.4443359, -174.6592560, 174.7268524
6: -119.7718124, 91.5253830, -119.8382263, 91.6494446, -211.4212646, 211.3636169
7: -102.3201599, 82.1205368, -102.4251251, 82.1728973, -184.4930267, 184.5456543
8: -107.3038330, 106.9020844, -107.4468994, 106.9559860, -214.2597961, 214.3489838
9: -82.8174438, 88.1460114, -82.8850708, 88.2113724, -171.0288086, 171.0310669
10: -122.9241180, 114.6170425, -123.0000153, 114.7142181, -237.6383362, 237.6170654
11: -123.0660400, 70.4828339, -123.1623306, 70.6079865, -193.6739960, 193.6451416
12: -120.2699356, 118.8007355, -120.3430557, 119.1533127, -239.4232483, 239.1437988
13: -129.5948944, 133.3258667, -129.6569519, 133.5224915, -263.1173706, 262.9828186
14: -189.5117493, 119.2154617, -189.6044006, 119.5260620, -309.0377808, 308.8198547
15: -91.3851776, 83.6129913, -91.5115967, 83.6814804, -175.0666504, 175.1245880
16: -128.6837616, 85.9889526, -128.7828369, 86.0426941, -214.7264404, 214.7717896
17: -187.7438354, 120.7224045, -187.8340912, 121.0605545, -308.8043823, 308.5564880
18: -124.4546890, 104.6576233, -124.5387039, 104.7306213, -229.1853027, 229.1963196
19: -90.4959717, 45.7087097, -90.5717316, 45.7276878, -136.2236633, 136.2804413
20: -84.9442749, 61.0196877, -85.0046844, 61.0762787, -146.0205536, 146.0243683
21: -114.2711792, 57.6654587, -114.3486557, 57.6977386, -171.9689178, 172.0141144
22: -120.5441971, 68.9448395, -120.6229095, 69.0066452, -189.5507965, 189.5677490
23: -90.4192734, 65.3167267, -90.4810257, 65.3483429, -155.7676086, 155.7977448
24: -115.3683777, 67.8023148, -115.4811096, 67.8302765, -183.1986542, 183.2834167
25: -97.8101730, 70.5343246, -97.8838501, 70.5792389, -168.3894043, 168.4181519
26: -134.0042114, 110.7619171, -134.0879211, 110.8921509, -244.8963623, 244.8498383
27: -122.3532639, 86.1951447, -122.4706497, 86.2371826, -208.5904236, 208.6658020
28: -89.8415375, 73.9924927, -89.9005432, 74.0325623, -163.8740997, 163.8930206
29: -128.4714508, 65.9165192, -128.5354614, 66.0214996, -194.4929504, 194.4519653
30: -114.9660568, 89.7126389, -115.0331879, 89.8089905, -204.7750549, 204.7458191
31: -117.5920715, 62.1937561, -117.6897125, 62.2215195, -179.8135986, 179.8834686
32: -122.0090561, 88.3287582, -122.0679092, 88.5040894, -210.5131226, 210.3966675
33: -153.0140381, 106.6683502, -153.2077637, 106.7269135, -259.7409058, 259.8760986
34: -127.2981186, 88.4586487, -127.3683472, 88.5126801, -215.8107910, 215.8269958
35: -124.3865280, 86.3766708, -124.4788818, 86.4164124, -210.8029327, 210.8555450
36: -128.7757263, 96.0787659, -128.8323364, 96.1805725, -224.9562988, 224.9110718
37: -175.5478516, 93.6099472, -175.6847534, 93.6519165, -269.1997681, 269.2947083
38: -154.4918823, 118.7677994, -154.5706024, 118.8484497, -273.3403320, 273.3383789
39: -170.2432861, 110.9598770, -170.3719940, 110.9965439, -281.2398376, 281.3318481
40: -142.6728058, 94.5067749, -142.7923279, 94.5461426, -237.2189484, 237.2990723
41: -122.1452942, 91.9150085, -122.2313385, 91.9633789, -214.1086731, 214.1463165
42: -90.3281097, 80.4803543, -90.3874817, 80.5608368, -170.8889465, 170.8678131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 1731
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
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 634
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
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1586
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
type: B, layer: 1, pos: 663
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
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 755
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1704
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
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1540
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
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1025
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
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 733
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
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1083
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
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1600
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
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1079
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
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1039
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
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1778
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
type: B, layer: 1, pos: 1081
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
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1487

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9555454, upper bound: 151.9983429
time: 271.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9555455, upper bound: 151.9983429
time: 173.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -153.8559265, 91.1313324, -153.6676331, 90.7571411, -244.6130676, 244.7989655
1: -79.4385910, 71.3770599, -79.2862244, 71.1116409, -150.5502319, 150.6632843
2: -75.0335236, 75.0486755, -74.9336243, 74.6762085, -149.7097321, 149.9822998
3: -82.8251190, 88.5652924, -82.7401352, 88.1645126, -170.9896240, 171.3054199
4: -86.8187943, 87.0413055, -86.7151871, 86.4990692, -173.3178711, 173.7565002
5: -85.4899139, 89.9447021, -85.4032135, 89.4679413, -174.9578552, 175.3479004
6: -120.2273712, 91.8121185, -119.8562927, 91.6863403, -211.9137115, 211.6683960
7: -102.6376572, 82.4202728, -102.4573975, 82.1915359, -184.8291931, 184.8776703
8: -107.6402283, 107.4275665, -107.5018158, 106.9744492, -214.6146851, 214.9293671
9: -83.1029663, 88.4049454, -82.9085922, 88.2295990, -171.3325653, 171.3135376
10: -123.5137100, 114.9421234, -123.0281372, 114.7412872, -238.2550049, 237.9702606
11: -123.8844681, 70.6663208, -123.1951065, 70.6107864, -194.4952393, 193.8614197
12: -121.2719803, 119.4372025, -120.3700104, 119.2975006, -240.5694427, 239.8072205
13: -130.0427246, 133.7823029, -129.6756287, 133.5982056, -263.6408997, 263.4579468
14: -190.3708954, 119.7259521, -189.6350098, 119.6572876, -310.0281982, 309.3609619
15: -91.6454315, 84.1072540, -91.5191422, 83.7039490, -175.3493805, 175.6263885
16: -129.2376709, 86.2355728, -128.8158264, 86.0582199, -215.2958832, 215.0513916
17: -188.8933716, 121.3007278, -187.8659973, 121.1950378, -310.0884094, 309.1666870
18: -124.8036957, 104.8871307, -124.5599365, 104.7550583, -229.5587463, 229.4470673
19: -90.9148178, 45.9066467, -90.5991516, 45.7331123, -136.6479187, 136.5057983
20: -85.2930908, 61.1697083, -85.0255585, 61.0947762, -146.3878632, 146.1952667
21: -114.8519669, 57.8202705, -114.3759079, 57.7087097, -172.5606689, 172.1961823
22: -120.9047546, 69.2331772, -120.6477509, 69.0278473, -189.9325867, 189.8809204
23: -90.7856979, 65.5266418, -90.5033188, 65.3584442, -156.1441345, 156.0299377
24: -115.6581421, 68.0684052, -115.5187073, 67.8390350, -183.4971771, 183.5871124
25: -98.0973587, 70.7984467, -97.9051361, 70.5928726, -168.6902313, 168.7035675
26: -134.6406860, 111.0670624, -134.1175232, 110.9391022, -245.5797882, 245.1845856
27: -122.6594543, 86.3944855, -122.4986496, 86.2496185, -208.9090729, 208.8931274
28: -90.1531982, 74.1359177, -89.9219589, 74.0467224, -164.1999207, 164.0578766
29: -128.9217224, 66.1635742, -128.5520325, 66.0650177, -194.9867401, 194.7156067
30: -115.2980042, 89.9425278, -115.0531616, 89.8222656, -205.1202698, 204.9956665
31: -118.0035324, 62.5250854, -117.7254944, 62.2292824, -180.2328186, 180.2505798
32: -122.4336472, 88.6596375, -122.0839462, 88.5716095, -211.0052490, 210.7435913
33: -153.4614716, 107.2040787, -153.2815704, 106.7444229, -260.2059021, 260.4856567
34: -127.5385895, 88.8736191, -127.3932190, 88.5290070, -216.0675964, 216.2668152
35: -124.6527328, 86.6659393, -124.5089264, 86.4274216, -211.0801544, 211.1748657
36: -129.1891022, 96.3449097, -128.8518219, 96.2228851, -225.4119568, 225.1967010
37: -176.0040588, 93.8485031, -175.7314758, 93.6657562, -269.6697998, 269.5799866
38: -154.9717255, 119.0351410, -154.5966339, 118.8769760, -273.8486938, 273.6317749
39: -170.6786499, 111.3346558, -170.4160919, 111.0045242, -281.6831665, 281.7507324
40: -143.0248566, 94.8417969, -142.8309937, 94.5564651, -237.5813293, 237.6727905
41: -122.5614090, 92.1179657, -122.2604828, 91.9788437, -214.5402527, 214.3784485
42: -90.8173828, 80.6997833, -90.4067688, 80.5796814, -171.3970642, 171.1065521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=628, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 651
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
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 602
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
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1590
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
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 634
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
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1125
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
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 624
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9530951, upper bound: 152.0011699
time: 164.20 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9530952, upper bound: 152.0011699
time: 224.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 391.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 391.02
Output dim: 11, lower bound: -151.9555454, upper bound: 151.9983429
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 391.02
Output dim: 11, lower bound: -151.9555455, upper bound: 151.9983429
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 391.02
Output dim: 11, lower bound: -151.9530951, upper bound: 152.0011699
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 391.02
Output dim: 11, lower bound: -151.9530952, upper bound: 152.0011699

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -153.3380737, 90.6431503, -153.1708984, 90.4508514, -243.7889252, 243.8140564
1: -79.1686707, 71.0410614, -79.0930634, 70.9092255, -150.0778961, 150.1341248
2: -74.6107254, 74.5845184, -74.4838104, 74.3886185, -148.9993439, 149.0683289
3: -82.4091187, 88.0357666, -82.2484589, 87.8105545, -170.2196655, 170.2842102
4: -86.3483734, 86.3921661, -86.2645416, 86.1687164, -172.5170898, 172.6567078
5: -85.0611801, 89.3231583, -84.9490509, 89.0819778, -174.1431580, 174.2721863
6: -119.7035370, 91.3978729, -119.5127869, 91.2789993, -210.9825439, 210.9106598
7: -102.1942520, 82.0698013, -102.0613327, 81.9309006, -184.1251221, 184.1311340
8: -107.1694183, 106.8554077, -107.0815887, 106.7109222, -213.8803406, 213.9369812
9: -82.6884918, 88.0843277, -82.5227432, 87.8876114, -170.5761108, 170.6070709
10: -122.8005219, 114.5319366, -122.5855026, 114.3154907, -237.1160126, 237.1174316
11: -122.9854660, 70.3226318, -122.6172714, 70.2004013, -193.1858673, 192.9398804
12: -120.1971970, 118.4974518, -119.8188095, 118.3802109, -238.5774078, 238.3162537
13: -129.4769897, 133.2310028, -129.3213806, 133.1539001, -262.6308899, 262.5523682
14: -189.3901672, 118.9307556, -188.9770050, 118.8169632, -308.2071228, 307.9077759
15: -91.2342529, 83.5554352, -91.0950546, 83.4033051, -174.6375580, 174.6504669
16: -128.5919800, 85.9345245, -128.4460602, 85.8047028, -214.3966522, 214.3805847
17: -187.6365356, 120.3459625, -187.0014038, 120.1272049, -307.7637329, 307.3473206
18: -124.3772278, 104.4301147, -124.1015472, 104.1659088, -228.5431366, 228.5316620
19: -90.4261169, 45.6763954, -90.1839447, 45.6189079, -136.0450134, 135.8603363
20: -84.8814087, 60.9568710, -84.6935425, 60.8887329, -145.7701416, 145.6504211
21: -114.1930237, 57.5924263, -113.8569794, 57.4863396, -171.6793365, 171.4494019
22: -120.4668427, 68.8355103, -120.1885681, 68.6945343, -189.1613770, 189.0240784
23: -90.3579712, 65.2890930, -90.1618042, 65.2542267, -155.6121979, 155.4508972
24: -115.2923584, 67.7497559, -115.1576233, 67.6699219, -182.9622803, 182.9073639
25: -97.7381973, 70.4885406, -97.6014481, 70.4261017, -168.1642761, 168.0899963
26: -133.9242859, 110.5155945, -133.5050354, 110.2716599, -244.1959534, 244.0206299
27: -122.2628098, 86.0291443, -121.9772110, 85.8133087, -208.0761108, 208.0063324
28: -89.7774811, 73.8884811, -89.5264893, 73.7566528, -163.5341187, 163.4149780
29: -128.3963470, 65.7570572, -128.0121765, 65.6161194, -194.0124512, 193.7692261
30: -114.8986969, 89.5583496, -114.6565857, 89.4070282, -204.3057251, 204.2149353
31: -117.5054474, 62.1564598, -117.3043213, 62.0833282, -179.5887604, 179.4607544
32: -121.9387665, 88.1859818, -121.7961807, 88.1081161, -210.0468750, 209.9821472
33: -152.8000946, 106.6033173, -152.6374512, 106.2564392, -259.0565186, 259.2407837
34: -127.2156982, 88.4063187, -127.0995026, 88.2363815, -215.4520721, 215.5058136
35: -124.2795105, 86.3383942, -124.1612473, 86.2024155, -210.4819336, 210.4996338
36: -128.7129364, 95.9597473, -128.5137939, 95.8292084, -224.5421448, 224.4735413
37: -175.4265900, 93.5485229, -175.2636566, 93.4404144, -268.8670044, 268.8121948
38: -154.4058838, 118.6824951, -154.2161560, 118.5324631, -272.9383545, 272.8986206
39: -170.1019287, 110.9181442, -169.9722137, 110.6192169, -280.7211304, 280.8903503
40: -142.5786438, 94.4665985, -142.4902649, 94.3492889, -236.9279327, 236.9568634
41: -122.0609131, 91.8279419, -121.9387512, 91.6879578, -213.7488708, 213.7666931
42: -90.2524872, 80.4149780, -90.1067200, 80.3397827, -170.5922546, 170.5216980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1753
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
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1559
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
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 645
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
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1720
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
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1562
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
type: A, layer: 1, pos: 907
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
type: A, layer: 1, pos: 663
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
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1643
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
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 736
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
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1155
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
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1125
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
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 635
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
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
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
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1723
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
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 537
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
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1601
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
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 968
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
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1487

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9257123, upper bound: 151.9481548
time: 209.82 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9244109, upper bound: 151.9668947
time: 222.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -153.4844818, 90.6852493, -153.5949707, 90.7268524, -244.2113342, 244.2802124
1: -79.2192764, 71.0690765, -79.2597733, 71.0912781, -150.3105469, 150.3288574
2: -74.7606506, 74.6197052, -74.8658295, 74.6520081, -149.4126587, 149.4855347
3: -82.5720291, 88.0842133, -82.6711884, 88.1332321, -170.7052460, 170.7553864
4: -86.4926758, 86.4412537, -86.6337509, 86.4742966, -172.9669495, 173.0750122
5: -85.2044067, 89.3721161, -85.3203964, 89.4314499, -174.6358337, 174.6925049
6: -119.7667923, 91.5031052, -119.8241501, 91.5917053, -211.3584900, 211.3272552
7: -102.3117294, 82.1152649, -102.4013596, 82.1581116, -184.4698486, 184.5166321
8: -107.2951431, 106.8970490, -107.4226379, 106.9417114, -214.2368469, 214.3196869
9: -82.8103180, 88.1415329, -82.8652115, 88.1989136, -171.0092316, 171.0067444
10: -122.9115753, 114.6109924, -122.9642944, 114.6979065, -237.6094666, 237.5752869
11: -123.0604858, 70.4673920, -123.1468811, 70.5686493, -193.6291351, 193.6142731
12: -120.2640381, 118.7853012, -120.3265762, 119.1108246, -239.3748627, 239.1118774
13: -129.5703430, 133.3189087, -129.5868225, 133.5032043, -263.0735474, 262.9057312
14: -189.5022583, 119.2024612, -189.5779724, 119.4895325, -308.9917603, 308.7804260
15: -91.3660660, 83.6086044, -91.4595184, 83.6691818, -175.0352325, 175.0681000
16: -128.6765900, 85.9831772, -128.7628937, 86.0264282, -214.7029724, 214.7460632
17: -187.7352905, 120.7052307, -187.8100891, 121.0128784, -308.7481689, 308.5153198
18: -124.4480743, 104.6390991, -124.5200272, 104.6789322, -229.1269836, 229.1591187
19: -90.4908752, 45.7050552, -90.5576553, 45.7172318, -136.2080994, 136.2627106
20: -84.9397888, 61.0102577, -84.9922638, 61.0494041, -145.9891968, 146.0025177
21: -114.2657547, 57.6564522, -114.3335724, 57.6732368, -171.9389954, 171.9900208
22: -120.5365067, 68.9389572, -120.6013336, 68.9901276, -189.5266418, 189.5402527
23: -90.4148102, 65.3135376, -90.4685822, 65.3392792, -155.7540894, 155.7821045
24: -115.3608704, 67.7938919, -115.4602814, 67.8064117, -183.1672668, 183.2541656
25: -97.8028183, 70.5306549, -97.8631668, 70.5689774, -168.3717804, 168.3938293
26: -133.9969940, 110.7496719, -134.0679321, 110.8578644, -244.8548584, 244.8175964
27: -122.3466721, 86.1804962, -122.4525223, 86.2017136, -208.5483704, 208.6330109
28: -89.8372879, 73.9859161, -89.8889160, 74.0178146, -163.8550873, 163.8748169
29: -128.4637451, 65.9082718, -128.5136261, 65.9981766, -194.4619141, 194.4218903
30: -114.9601288, 89.6986465, -115.0165100, 89.7703018, -204.7304077, 204.7151489
31: -117.5858383, 62.1909370, -117.6725006, 62.2136803, -179.7995148, 179.8634186
32: -122.0036697, 88.3123398, -122.0528336, 88.4579926, -210.4616699, 210.3651581
33: -153.0003510, 106.6642227, -153.1698608, 106.7154236, -259.7157593, 259.8340759
34: -127.2911606, 88.4548721, -127.3490448, 88.5020981, -215.7932587, 215.8039093
35: -124.3724442, 86.3736572, -124.4400787, 86.4079437, -210.7803802, 210.8137360
36: -128.7712097, 96.0712051, -128.8199768, 96.1591797, -224.9303894, 224.8911743
37: -175.5381317, 93.6014709, -175.6576691, 93.6277618, -269.1658936, 269.2591248
38: -154.4852905, 118.7573547, -154.5526276, 118.8185425, -273.3038330, 273.3099670
39: -170.2319183, 110.9566956, -170.3410950, 110.9877014, -281.2196045, 281.2977905
40: -142.6655273, 94.5020676, -142.7721405, 94.5329514, -237.1984863, 237.2742004
41: -122.1399078, 91.8999329, -122.2164764, 91.9209900, -214.0608978, 214.1164093
42: -90.3237534, 80.4738159, -90.3755035, 80.5429840, -170.8667297, 170.8493042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 855
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
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1062
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9257123, upper bound: 151.9481548
time: 194.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9725534, upper bound: 151.9668947
time: 202.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -153.6984253, 91.0855560, -153.2120972, 90.4696121, -244.1680298, 244.2976532
1: -79.3835602, 71.3457565, -79.1062775, 70.9197311, -150.3032837, 150.4520264
2: -74.8758316, 75.0101624, -74.5290985, 74.4028625, -149.2786713, 149.5392609
3: -82.6543427, 88.5128708, -82.2935486, 87.8298187, -170.4841614, 170.8063965
4: -86.6671066, 86.9883423, -86.3241959, 86.1819077, -172.8490143, 173.3125305
5: -85.3366089, 89.8913879, -85.0022583, 89.1057816, -174.4423828, 174.8936462
6: -120.1597595, 91.6841583, -119.5311127, 91.3154449, -211.4751892, 211.2152405
7: -102.5120544, 82.3703384, -102.0935135, 81.9498138, -184.4618683, 184.4638367
8: -107.5061264, 107.3810959, -107.1365356, 106.7295990, -214.2357178, 214.5176392
9: -82.9756165, 88.3427734, -82.5459366, 87.9058609, -170.8814697, 170.8887024
10: -123.3918152, 114.8563156, -122.6132202, 114.3425903, -237.7344055, 237.4695282
11: -123.8056488, 70.5068665, -122.6504211, 70.2034302, -194.0090637, 193.1572876
12: -121.1998062, 119.1346588, -119.8459702, 118.5244064, -239.7241974, 238.9805908
13: -129.9250488, 133.6865997, -129.3400574, 133.2297821, -263.1548462, 263.0266418
14: -190.2497864, 119.4412155, -189.0079193, 118.9481277, -309.1979065, 308.4491272
15: -91.4951782, 84.0497894, -91.1028061, 83.4259262, -174.9210968, 175.1525726
16: -129.1463470, 86.1810455, -128.4793091, 85.8204956, -214.9668427, 214.6603394
17: -188.7866211, 120.9243240, -187.0335846, 120.2614365, -309.0480347, 307.9578857
18: -124.7255020, 104.6599731, -124.1224594, 104.1901703, -228.9156494, 228.7824402
19: -90.8462067, 45.8746223, -90.2116547, 45.6242905, -136.4704895, 136.0862732
20: -85.2306976, 61.1068344, -84.7144852, 60.9071999, -146.1378937, 145.8213196
21: -114.7750015, 57.7477074, -113.8845978, 57.4972610, -172.2722626, 171.6323090
22: -120.8273621, 69.1242676, -120.2135315, 68.7156067, -189.5429535, 189.3377838
23: -90.7251663, 65.4991150, -90.1843262, 65.2643127, -155.9894714, 155.6834412
24: -115.5818024, 68.0161743, -115.1954422, 67.6786270, -183.2604370, 183.2115936
25: -98.0252686, 70.7530518, -97.6228790, 70.4396439, -168.4649048, 168.3759308
26: -134.5612640, 110.8212128, -133.5349731, 110.3185425, -244.8798065, 244.3561859
27: -122.5681992, 86.2297516, -122.0051575, 85.8253632, -208.3935547, 208.2349091
28: -90.0893173, 74.0325165, -89.5481415, 73.7706985, -163.8600159, 163.5806580
29: -128.8469086, 66.0041199, -128.0288239, 65.6595154, -194.5064240, 194.0329285
30: -115.2307739, 89.7884674, -114.6768188, 89.4203949, -204.6511688, 204.4652863
31: -117.9169922, 62.4880142, -117.3404694, 62.0911217, -180.0081177, 179.8284912
32: -122.3638382, 88.5171814, -121.8122940, 88.1755447, -210.5393372, 210.3294678
33: -153.2476654, 107.1394653, -152.7111664, 106.2740860, -259.5217590, 259.8506165
34: -127.4567413, 88.8212814, -127.1243515, 88.2528458, -215.7095795, 215.9456329
35: -124.5457916, 86.6279068, -124.1912231, 86.2136078, -210.7593689, 210.8191223
36: -129.1265869, 96.2257233, -128.5334015, 95.8713989, -224.9979858, 224.7591248
37: -175.8833618, 93.7870483, -175.3104858, 93.4542465, -269.3375854, 269.0975037
38: -154.8862000, 118.9492645, -154.2423401, 118.5607910, -273.4469910, 273.1915894
39: -170.5374146, 111.2932587, -170.0161591, 110.6273270, -281.1647339, 281.3094177
40: -142.9304504, 94.8014832, -142.5289612, 94.3596725, -237.2900848, 237.3304443
41: -122.4778519, 92.0310135, -121.9680328, 91.7034073, -214.1812592, 213.9990540
42: -90.7422791, 80.6341324, -90.1260834, 80.3585663, -171.1008301, 170.7602234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=627, inp2_unstable=627, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=847, inp2_unstable=847, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1753
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
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 621
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
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1717
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
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 965
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
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1613
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
type: A, layer: 1, pos: 663
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
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1643
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
type: A, layer: 1, pos: 887
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
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1037
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
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 688
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
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1233
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
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 967
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
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1749
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1601
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
Output dim: 11, lower bound: -151.9233859, upper bound: 151.9514154
time: 2106.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9220276, upper bound: 151.9696538
time: 180.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2289.23 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9257123, upper bound: 151.9481548
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9244109, upper bound: 151.9668947
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9257123, upper bound: 151.9481548
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9725534, upper bound: 151.9668947
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9233859, upper bound: 151.9514154
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2289.23
Output dim: 11, lower bound: -151.9220276, upper bound: 151.9696538
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2289.23
Output dim: 11, lower bound: -151.9530952, upper bound: 152.0011699
Binary search (step 2): status=Status.UNKNOWN, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=193.89141845703125
rel_dist={11: [-152.0594572987635, 152.05945742807847]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.015625
execution time: 14836.37 seconds
