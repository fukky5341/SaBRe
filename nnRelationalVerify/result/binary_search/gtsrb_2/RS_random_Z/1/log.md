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
execution time: IAR + LP analysis = 2.83 + 179.82 = 182.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -156.8589948, upper bound: 156.8589950


# Binary Search by BASE starts (time budget: 17817.35 seconds, max iter: 100)

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
Binary search time: 876.42 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.015625


# Relational Split (RS_random_Z) starts
Time budget: 16940.93 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1676

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5521270, upper bound: 154.5519357
time: 211.12 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5519354, upper bound: 154.5521272
time: 189.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 400.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 400.25
Output dim: 11, lower bound: -154.5521270, upper bound: 154.5519357
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 400.25
Output dim: 11, lower bound: -154.5519354, upper bound: 154.5521272

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 596

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 688

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5521272, upper bound: 154.5511126
time: 384.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5513122, upper bound: 154.5519357
time: 514.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 532

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5370558, upper bound: 154.5513266
time: 753.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5511395, upper bound: 154.5372674
time: 172.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 928.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 928.20
Output dim: 11, lower bound: -154.5521272, upper bound: 154.5511126
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 928.20
Output dim: 11, lower bound: -154.5513122, upper bound: 154.5519357
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 928.20
Output dim: 11, lower bound: -154.5370558, upper bound: 154.5513266
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 928.20
Output dim: 11, lower bound: -154.5511395, upper bound: 154.5372674

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1036

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 659

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5450978, upper bound: 154.5506173
time: 195.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5516325, upper bound: 154.5441568
time: 921.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 652

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5501043, upper bound: 154.5515876
time: 190.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -154.5510180, upper bound: 154.5508930
time: 347.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 539.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 539.86
Output dim: 11, lower bound: -154.5450978, upper bound: 154.5506173
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 539.86
Output dim: 11, lower bound: -154.5516325, upper bound: 154.5441568
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 539.86
Output dim: 11, lower bound: -154.5501043, upper bound: 154.5515876
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 539.86
Output dim: 11, lower bound: -154.5510180, upper bound: 154.5508930
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 539.86
Output dim: 11, lower bound: -154.5370558, upper bound: 154.5513266
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 539.86
Output dim: 11, lower bound: -154.5511395, upper bound: 154.5372674
Binary search (step 0): status=Status.UNKNOWN, k_low=5, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=193.89141845703125
rel_dist={11: [-154.55252351539193, 154.55252373789625]}

## Binary search (step 1) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 840

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0064365, upper bound: 153.0131716
time: 186.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0131715, upper bound: 153.0064367
time: 647.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 833.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 833.97
Output dim: 11, lower bound: -153.0064365, upper bound: 153.0131716
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 833.97
Output dim: 11, lower bound: -153.0131715, upper bound: 153.0064367

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 968

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0063520, upper bound: 153.0080340
time: 177.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0012630, upper bound: 153.0130869
time: 227.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1035

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1542

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0071239, upper bound: 153.0061974
time: 1240.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0129322, upper bound: 153.0004037
time: 216.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1459.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1459.05
Output dim: 11, lower bound: -153.0063520, upper bound: 153.0080340
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1459.05
Output dim: 11, lower bound: -153.0012630, upper bound: 153.0130869
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1459.05
Output dim: 11, lower bound: -153.0071239, upper bound: 153.0061974
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1459.05
Output dim: 11, lower bound: -153.0129322, upper bound: 153.0004037

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1742

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 651

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9973796, upper bound: 153.0071300
time: 207.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0054913, upper bound: 152.9990774
time: 192.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 537

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9965587, upper bound: 153.0083197
time: 229.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9965587, upper bound: 153.0130869
time: 197.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1549

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 530

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -153.0067576, upper bound: 152.9832778
time: 245.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.9841753, upper bound: 153.0058312
time: 172.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 420.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -152.9973796, upper bound: 153.0071300
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -153.0054913, upper bound: 152.9990774
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -152.9965587, upper bound: 153.0083197
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -152.9965587, upper bound: 153.0130869
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -153.0067576, upper bound: 152.9832778
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 420.46
Output dim: 11, lower bound: -152.9841753, upper bound: 153.0058312
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 420.46
Output dim: 11, lower bound: -153.0129322, upper bound: 153.0004037
Binary search (step 1): status=Status.UNKNOWN, k_low=5, k_high=7, k_mid=6, eps_mid=0.0234375, abs_max=193.89141845703125
rel_dist={11: [-153.0152404825352, 153.01524066488776]}

## Binary search (step 2) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1055

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 746

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -152.0045632, upper bound: 151.9936361
time: 178.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9936359, upper bound: 152.0045634
time: 196.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 375.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 375.67
Output dim: 11, lower bound: -152.0045632, upper bound: 151.9936361
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 375.67
Output dim: 11, lower bound: -151.9936359, upper bound: 152.0045634

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1037

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9904045, upper bound: 151.9709518
time: 227.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9822397, upper bound: 151.9904046
time: 340.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -153.6976929, 90.7650757, -153.6976929, 90.7650757, -244.4627686, 244.4627686
1: -79.2997437, 71.1168976, -79.2997437, 71.1168976, -150.4166412, 150.4166412
2: -74.9523544, 74.6830826, -74.9523544, 74.6830826, -149.6354370, 149.6354370
3: -82.7562866, 88.1763611, -82.7562866, 88.1763611, -170.9326477, 170.9326477
4: -86.7386017, 86.5084610, -86.7386017, 86.5084610, -173.2470703, 173.2470703
5: -85.4221344, 89.4804077, -85.4221344, 89.4804077, -174.9025421, 174.9025421
6: -119.8750687, 91.7171707, -119.8750687, 91.7171707, -211.5922241, 211.5922241
7: -102.4825897, 82.2010651, -102.4825897, 82.2010651, -184.6836548, 184.6836548
8: -107.5238342, 106.9852600, -107.5238342, 106.9852600, -214.5090637, 214.5090637
9: -82.9216461, 88.2472992, -82.9216461, 88.2472992, -171.1689453, 171.1689453
10: -123.0418167, 114.7664795, -123.0418167, 114.7664795, -237.8082886, 237.8082886
11: -123.2149811, 70.6764374, -123.2149811, 70.6764374, -193.8914185, 193.8914185
12: -120.3820953, 119.3412552, -120.3820953, 119.3412552, -239.7233276, 239.7233276
13: -129.6905365, 133.6348267, -129.6905365, 133.6348267, -263.3253174, 263.3253174
14: -189.6543274, 119.6912918, -189.6543274, 119.6912918, -309.3455811, 309.3455811
15: -91.5800781, 83.7190857, -91.5800781, 83.7190857, -175.2991333, 175.2991333
16: -128.8368073, 86.0722809, -128.8368073, 86.0722809, -214.9090881, 214.9090881
17: -187.8823853, 121.2422180, -187.8823853, 121.2422180, -309.1246033, 309.1246033
18: -124.5863342, 104.7705078, -124.5863342, 104.7705078, -229.3568420, 229.3568420
19: -90.6129379, 45.7380524, -90.6129379, 45.7380524, -136.3509827, 136.3509827
20: -85.0376434, 61.1067200, -85.0376434, 61.1067200, -146.1443634, 146.1443634
21: -114.3909760, 57.7151794, -114.3909760, 57.7151794, -172.1061554, 172.1061401
22: -120.6659775, 69.0411530, -120.6659775, 69.0411530, -189.7071228, 189.7071228
23: -90.5145111, 65.3655319, -90.5145111, 65.3655319, -155.8800354, 155.8800354
24: -115.5423126, 67.8453522, -115.5423126, 67.8453522, -183.3876648, 183.3876648
25: -97.9238129, 70.6036148, -97.9238129, 70.6036148, -168.5274200, 168.5274353
26: -134.1332245, 110.9630814, -134.1332245, 110.9630814, -245.0962830, 245.0962830
27: -122.5359192, 86.2600250, -122.5359192, 86.2600250, -208.7958984, 208.7958984
28: -89.9322357, 74.0543976, -89.9322357, 74.0543976, -163.9866028, 163.9866180
29: -128.5706787, 66.0791016, -128.5706787, 66.0791016, -194.6497803, 194.6497650
30: -115.0701752, 89.8611221, -115.0701752, 89.8611221, -204.9313049, 204.9313049
31: -117.7433777, 62.2366028, -117.7433777, 62.2366028, -179.9799805, 179.9799805
32: -122.1000977, 88.5980301, -122.1000977, 88.5980301, -210.6981201, 210.6981201
33: -153.3112793, 106.7589111, -153.3112793, 106.7589111, -260.0701904, 260.0701904
34: -127.4062729, 88.5423737, -127.4062729, 88.5423737, -215.9486389, 215.9486389
35: -124.5285568, 86.4385300, -124.5285568, 86.4385300, -210.9670868, 210.9670868
36: -128.8628235, 96.2365112, -128.8628235, 96.2365112, -225.0993347, 225.0993347
37: -175.7586212, 93.6746063, -175.7586212, 93.6746063, -269.4331970, 269.4332275
38: -154.6128235, 118.8931351, -154.6128235, 118.8931351, -273.5059204, 273.5059509
39: -170.4415894, 111.0169754, -170.4415894, 111.0169754, -281.4585571, 281.4585571
40: -142.8583374, 94.5674896, -142.8583374, 94.5674896, -237.4258118, 237.4258118
41: -122.2781219, 91.9895172, -122.2781219, 91.9895172, -214.2676392, 214.2676392
42: -90.4201202, 80.6052628, -90.4201202, 80.6052628, -171.0253906, 171.0253906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=628, inp2_unstable=628, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=848, inp2_unstable=848, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=35, inp2_unstable=35, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1079

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1677

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9897139, upper bound: 151.9897141
time: 7092.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -151.9904183, upper bound: 152.0006258
time: 169.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7264.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7264.09
Output dim: 11, lower bound: -151.9904045, upper bound: 151.9709518
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7264.09
Output dim: 11, lower bound: -151.9822397, upper bound: 151.9904046
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7264.09
Output dim: 11, lower bound: -151.9897139, upper bound: 151.9897141
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7264.09
Output dim: 11, lower bound: -151.9904183, upper bound: 152.0006258
Binary search (step 2): status=Status.UNKNOWN, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=193.89141845703125
rel_dist={11: [-152.0594572987635, 152.05945742807847]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.015625
execution time: 16884.03 seconds
