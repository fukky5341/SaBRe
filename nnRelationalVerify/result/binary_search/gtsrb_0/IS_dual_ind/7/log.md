## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 88.175177343
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052)
1: (-59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901)
2: (-49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462)
3: (-62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052)
4: (-55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501)
5: (-67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838)
6: (-69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731)
7: (-86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911)
8: (-64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949)
9: (-39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489)
10: (-77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949)
11: (-93.2068634, 14.3253803, -93.2068634, 14.3253803, -107.5322418, 107.5322418)
12: (-58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337)
13: (-64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540)
14: (-142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217)
15: (-56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481)
16: (-85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3501282, 115.3501282)
17: (-157.8650360, 22.9012489, -157.8650360, 22.9012489, -175.1945038, 175.1945038)
18: (-75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331)
19: (-60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232)
20: (-51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269)
21: (-78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912)
22: (-82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912)
23: (-52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010)
24: (-50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713)
25: (-46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737)
26: (-77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283)
27: (-73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169)
28: (-58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228)
29: (-91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255)
30: (-69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819)
31: (-66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800)
32: (-67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420)
33: (-58.9121552, 82.5250702, -58.9121552, 82.5250702, -141.4372253, 141.4372253)
34: (-60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.7704163, 129.7704163)
35: (-50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645)
36: (-60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887)
37: (-56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.8929443, 117.8929596)
38: (-75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558)
39: (-64.1901398, 88.0310287, -64.1901398, 88.0310287, -152.2211609, 152.2211609)
40: (-59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337)
41: (-51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757)
42: (-56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685)

## BASE Result
execution time: IAR + LP analysis = 3.13 + 54.69 = 57.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -96.1728623, upper bound: 96.1728623


# Binary Search by BASE starts (time budget: 17942.19 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=116.50578308105469
rel_dist={37: [-88.24876147168659, 88.24876147750362]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=115.81219482421875
rel_dist={37: [-83.42980060643012, 83.42980061641649]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=116.04339599609375
rel_dist={37: [-85.15018339085728, 85.15018339368487]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=116.27458190917969
rel_dist={37: [-86.7457354692693, 86.74573546640596]}

## Binary Search Result
Binary search time: 506.94 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 17435.25 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3607750, upper bound: 92.2613703
time: 139.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3607750, upper bound: 92.3607748
time: 145.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 285.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 285.74
Output dim: 37, lower bound: -92.3607750, upper bound: 92.2613703
IS_A2, status: Status.UNKNOWN, split count: 1, time: 285.74
Output dim: 37, lower bound: -92.3607750, upper bound: 92.3607748

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5766144, 36.7272797, -92.7802811, 36.9041595, -129.4807739, 129.5075684
1: -59.3741531, 29.7235413, -59.5148697, 29.8928299, -89.2669830, 89.2384109
2: -49.3999863, 30.7474213, -49.5429077, 30.8782902, -80.2782745, 80.2903290
3: -62.2667351, 31.1304169, -62.4098053, 31.2786102, -93.5453491, 93.5402222
4: -54.8909836, 43.5707092, -54.9914131, 43.6777840, -98.5687714, 98.5621185
5: -67.2087097, 36.8288574, -67.3472824, 36.9504166, -104.1591263, 104.1761398
6: -69.0129242, 41.5568008, -69.1043854, 41.6018448, -110.6147690, 110.6611862
7: -86.5544739, 27.0560989, -86.7408981, 27.2529602, -113.8074341, 113.7969971
8: -64.6697540, 51.6969833, -64.8798065, 51.9112549, -116.5810089, 116.5767899
9: -39.1258163, 35.8437691, -39.1837730, 35.9050598, -75.0308762, 75.0275421
10: -77.2260742, 47.7212410, -77.3258514, 47.8095322, -125.0356064, 125.0470886
11: -92.9816284, 14.2313957, -93.1657410, 14.3124256, -106.7347107, 106.8387909
12: -58.8658867, 51.9315338, -58.9842148, 52.0541649, -110.9200516, 110.9157486
13: -64.1183701, 66.7683258, -64.1710663, 66.8636398, -130.9820099, 130.9393921
14: -142.3578949, 20.9416828, -142.6149902, 21.1001911, -163.4580841, 163.5566711
15: -55.9344368, 46.6895332, -56.0442314, 46.7611160, -102.6955566, 102.7337646
16: -85.5276489, 29.5867386, -85.6360245, 29.6797905, -115.2074432, 115.2227631
17: -157.6787872, 22.7648430, -157.8322296, 22.8888702, -170.1320190, 170.1613770
18: -75.0799866, 48.5977325, -75.2375031, 48.7552795, -123.8352661, 123.8352356
19: -60.3920822, 14.7219152, -60.4535332, 14.7668648, -75.1589508, 75.1754456
20: -51.6937790, 18.7260647, -51.7784691, 18.7805271, -70.4743042, 70.5045319
21: -78.5317917, 18.6328545, -78.6202087, 18.6838932, -97.2156830, 97.2530670
22: -82.7314148, 24.7807159, -82.8434525, 24.8906078, -107.6220245, 107.6241684
23: -51.9689713, 24.5617599, -52.0309601, 24.5973969, -76.5663681, 76.5927200
24: -50.1737747, 33.0148849, -50.2776909, 33.0891190, -83.2628937, 83.2925720
25: -46.3452454, 35.6632195, -46.4810715, 35.8196869, -82.1649323, 82.1442871
26: -76.9218979, 47.6517982, -77.1318207, 47.7617455, -124.6836395, 124.7836151
27: -73.1140900, 30.4339638, -73.1874542, 30.4703140, -103.5844040, 103.6214142
28: -58.0445290, 31.2149391, -58.1240501, 31.2613964, -89.3059235, 89.3389893
29: -91.6434479, 15.8860188, -91.7454834, 15.9489250, -107.5923767, 107.6315002
30: -69.8655548, 37.1955566, -69.9683533, 37.2802887, -107.1458435, 107.1639099
31: -66.1632385, 18.6622314, -66.2685013, 18.7370262, -84.9002686, 84.9307327
32: -67.4447632, 42.2245445, -67.5149078, 42.2756119, -109.7203751, 109.7394562
33: -58.7477188, 82.2953339, -58.9011002, 82.4803314, -140.1516113, 140.1244049
34: -60.4687729, 68.9983368, -60.5968704, 69.1301422, -129.3390045, 129.3294678
35: -50.7373390, 76.7162170, -50.8142242, 76.7949371, -127.5322723, 127.5304413
36: -60.5996056, 67.1382904, -60.7377243, 67.2077484, -127.8073578, 127.8760147
37: -56.3634644, 61.1411667, -56.5719757, 61.2807198, -116.9335709, 117.0021896
38: -75.4035110, 82.5364304, -75.5154419, 82.5923538, -157.9958649, 158.0518799
39: -64.0875854, 87.8962097, -64.1786423, 88.0010681, -151.4566498, 151.4452820
40: -59.3607330, 57.8673897, -59.4794350, 57.9386330, -117.2993622, 117.3468246
41: -50.8831215, 44.4067078, -50.9756699, 44.4383850, -95.3215027, 95.3823776
42: -56.1530457, 40.2920837, -56.2213898, 40.3457413, -96.4987869, 96.5134735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2568645, upper bound: 92.2440007
time: 185.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2568645, upper bound: 92.2397352
time: 98.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -92.8168945, 36.9064636, -92.8269806, 36.9121208, -129.7290192, 129.7334442
1: -59.5403709, 29.8971672, -59.5488739, 29.9010124, -89.4413834, 89.4460449
2: -49.5681419, 30.8800659, -49.5766220, 30.8851204, -80.4532623, 80.4566879
3: -62.4353523, 31.2812347, -62.4454842, 31.2880211, -93.7233734, 93.7267151
4: -55.0073509, 43.6801758, -55.0142899, 43.6848602, -98.6922150, 98.6944656
5: -67.3706284, 36.9514656, -67.3801270, 36.9576530, -104.3282776, 104.3315887
6: -69.0903168, 41.6073341, -69.1245422, 41.6147308, -110.7050476, 110.7318726
7: -86.7659836, 27.2570705, -86.7885284, 27.2638626, -114.0298462, 114.0456009
8: -64.9211044, 51.9135361, -64.9305267, 51.9217682, -116.8428726, 116.8440628
9: -39.1908340, 35.9055328, -39.1947594, 35.9121933, -75.1030273, 75.1002960
10: -77.3383026, 47.8122673, -77.3452911, 47.8195038, -125.1578064, 125.1575623
11: -93.1560059, 14.3186169, -93.2068634, 14.3253803, -106.9149017, 106.9646454
12: -58.9852676, 52.0732651, -58.9932175, 52.0801201, -111.0653839, 111.0664825
13: -64.1752014, 66.8678436, -64.1798248, 66.8761292, -131.0513306, 131.0476685
14: -142.6594696, 21.1012630, -142.6708984, 21.1119289, -163.7713928, 163.7721558
15: -56.0562820, 46.7623138, -56.0640907, 46.7733612, -102.8296432, 102.8264008
16: -85.6339722, 29.6846275, -85.6600189, 29.6901054, -115.3240814, 115.3446503
17: -157.8517761, 22.8917389, -157.8650360, 22.9012489, -170.2370911, 170.3286285
18: -75.2397614, 48.7835312, -75.2505798, 48.7920532, -124.0318146, 124.0341110
19: -60.4546967, 14.7685261, -60.4632301, 14.7708912, -75.2255859, 75.2317581
20: -51.7836494, 18.7849102, -51.7888412, 18.7899876, -70.5736389, 70.5737534
21: -78.6214066, 18.6872406, -78.6373444, 18.6911507, -97.3125610, 97.3245850
22: -82.8503723, 24.8939972, -82.8589020, 24.9182854, -107.7686615, 107.7528992
23: -52.0333862, 24.6008911, -52.0391998, 24.6034031, -76.6367874, 76.6400909
24: -50.2813492, 33.0999718, -50.2867355, 33.1057358, -83.3870850, 83.3867035
25: -46.4842339, 35.8317833, -46.4892807, 35.8571930, -82.3414307, 82.3210602
26: -77.1396332, 47.7798233, -77.1492996, 47.7866287, -124.9262619, 124.9291229
27: -73.1921539, 30.4729614, -73.2002563, 30.4758568, -103.6680145, 103.6732178
28: -58.1272621, 31.2648582, -58.1312790, 31.2689457, -89.3962097, 89.3961334
29: -91.7566528, 15.9509907, -91.7652435, 15.9564838, -107.7131348, 107.7162323
30: -69.9721069, 37.2895775, -69.9780807, 37.2965050, -107.2686157, 107.2676544
31: -66.2725830, 18.7404671, -66.2767334, 18.7443485, -85.0169296, 85.0171967
32: -67.5052643, 42.2814713, -67.5284882, 42.2854538, -109.7907181, 109.8099594
33: -58.9039345, 82.5077591, -58.9121552, 82.5250702, -140.3601379, 140.3307343
34: -60.6011429, 69.1558380, -60.6091537, 69.1612701, -129.5097046, 129.4727783
35: -50.8169403, 76.8069000, -50.8236122, 76.8187485, -127.6356888, 127.6305084
36: -60.7409821, 67.2212677, -60.7492561, 67.2241364, -127.9651184, 127.9705200
37: -56.5746498, 61.3061066, -56.5913467, 61.3142395, -117.1824036, 117.1773224
38: -75.5191956, 82.5988159, -75.5316391, 82.6043243, -158.1235199, 158.1304626
39: -64.1811829, 88.0098114, -64.1901398, 88.0310287, -151.5708618, 151.5353546
40: -59.4822884, 57.9480896, -59.4948502, 57.9534836, -117.4357758, 117.4429398
41: -50.9663925, 44.4438477, -51.0001411, 44.4468384, -95.4132309, 95.4439850
42: -56.2136574, 40.3499680, -56.2370300, 40.3563385, -96.5699921, 96.5869980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2613703, upper bound: 92.3607750
time: 89.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2613703, upper bound: 92.3607750
time: 115.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 208.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 208.43
Output dim: 37, lower bound: -92.2568645, upper bound: 92.2440007
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 208.43
Output dim: 37, lower bound: -92.2568645, upper bound: 92.2397352
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 208.43
Output dim: 37, lower bound: -92.2613703, upper bound: 92.3607750
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 208.43
Output dim: 37, lower bound: -92.2613703, upper bound: 92.3607750

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -92.5336609, 36.7205124, -92.5173950, 36.7673950, -129.3010559, 129.2379150
1: -59.3439713, 29.7184563, -59.3443947, 29.7661209, -89.1100922, 89.0628510
2: -49.3767395, 30.7412491, -49.4158516, 30.7765350, -80.1532745, 80.1571045
3: -62.2566872, 31.1210518, -62.3482590, 31.1686401, -93.4253235, 93.4693146
4: -54.8578033, 43.5626831, -54.8126564, 43.5351562, -98.3929596, 98.3753357
5: -67.1897278, 36.8194427, -67.2430954, 36.8327103, -104.0224380, 104.0625381
6: -69.0069275, 41.5186310, -68.9798355, 41.3949394, -110.4018707, 110.4984665
7: -86.5239639, 27.0499420, -86.5802917, 27.1227436, -113.6467056, 113.6302338
8: -64.6268921, 51.6872635, -64.6575317, 51.7569275, -116.3838196, 116.3447952
9: -39.1124420, 35.8375168, -39.0830154, 35.7849350, -74.8973770, 74.9205322
10: -77.1770782, 47.7115746, -77.0705795, 47.6126251, -124.7897034, 124.7821503
11: -92.9591293, 14.2262478, -93.0298462, 14.2017937, -106.5980988, 106.6918182
12: -58.8543129, 51.9195404, -58.8753090, 51.9813118, -110.8356247, 110.7948456
13: -64.1085510, 66.7223206, -63.9819794, 66.6153641, -130.7239075, 130.7042999
14: -142.2827301, 20.9364548, -142.2099762, 20.9221992, -163.2049255, 163.1464233
15: -55.8991776, 46.6820374, -55.8538895, 46.6044312, -102.5036087, 102.5359268
16: -85.4919968, 29.5779305, -85.4292679, 29.5172844, -115.0092773, 115.0072021
17: -157.6047668, 22.7567253, -157.4434814, 22.6593628, -169.8270569, 169.7479706
18: -75.0485992, 48.5908623, -75.0626678, 48.6317749, -123.6803741, 123.6535339
19: -60.3769684, 14.7183037, -60.3480988, 14.7020588, -75.0790253, 75.0664062
20: -51.6842537, 18.7213516, -51.6840286, 18.7106972, -70.3949509, 70.4053802
21: -78.5101929, 18.6279831, -78.4823380, 18.5858059, -97.0960007, 97.1103210
22: -82.6970062, 24.7757206, -82.6503906, 24.7963943, -107.4934006, 107.4261093
23: -51.9567184, 24.5567646, -51.9362793, 24.5365143, -76.4932327, 76.4930420
24: -50.1570587, 33.0118408, -50.1278419, 33.0335693, -83.1906281, 83.1396790
25: -46.3334351, 35.6588669, -46.3386421, 35.7636681, -82.0971069, 81.9975128
26: -76.8965073, 47.6451569, -76.9649200, 47.6333389, -124.5298462, 124.6100769
27: -73.0950775, 30.4288445, -73.0680389, 30.3843517, -103.4794312, 103.4968872
28: -58.0337715, 31.2086010, -58.0440254, 31.2015324, -89.2353058, 89.2526245
29: -91.6035461, 15.8816967, -91.5247345, 15.8341198, -107.4376678, 107.4064331
30: -69.8485565, 37.1894073, -69.8481903, 37.1956406, -107.0441971, 107.0375977
31: -66.1528320, 18.6572609, -66.1624298, 18.6593342, -84.8121643, 84.8196869
32: -67.4342041, 42.1851273, -67.3600159, 42.0523453, -109.4865494, 109.5451431
33: -58.7343979, 82.2406235, -58.6475372, 82.2072678, -139.8629761, 139.8153992
34: -60.4585800, 68.9591522, -60.4335289, 68.9269714, -129.1195984, 129.1243591
35: -50.7278290, 76.6649780, -50.6048050, 76.5409470, -127.2687759, 127.2697830
36: -60.5911674, 67.0713806, -60.5271416, 66.8793335, -127.4705048, 127.5985260
37: -56.3473282, 61.1161537, -56.3224373, 61.1544495, -116.7901077, 116.7259216
38: -75.3908844, 82.4629440, -75.2423553, 82.2259903, -157.6168823, 157.7052917
39: -64.0718536, 87.8229523, -63.8759956, 87.6468201, -151.0834656, 151.0688171
40: -59.3475113, 57.8283386, -59.3162842, 57.7333145, -117.0808258, 117.1446228
41: -50.8754883, 44.3766212, -50.8439140, 44.2740707, -95.1495590, 95.2205353
42: -56.1466141, 40.2682800, -56.1295815, 40.2066116, -96.3532257, 96.3978577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2388283, upper bound: 92.1642379
time: 93.94 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2201642, upper bound: 92.2073375
time: 115.00 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.5766144, 36.7272797, -92.7731857, 36.9024582, -129.4790649, 129.5004578
1: -59.3741531, 29.7235413, -59.5093803, 29.8914948, -89.2656479, 89.2329254
2: -49.3999863, 30.7474213, -49.5404282, 30.8767815, -80.2767639, 80.2878494
3: -62.2667351, 31.1304169, -62.4032021, 31.2763634, -93.5430984, 93.5336151
4: -54.8909836, 43.5707092, -54.9850502, 43.6755486, -98.5665283, 98.5557556
5: -67.2087097, 36.8288574, -67.3426208, 36.9483109, -104.1570206, 104.1714783
6: -69.0129242, 41.5568008, -69.1026154, 41.5941887, -110.6071167, 110.6594162
7: -86.5544739, 27.0560989, -86.7364960, 27.2512074, -113.8056793, 113.7925949
8: -64.6697540, 51.6969833, -64.8718414, 51.9088860, -116.5786438, 116.5688248
9: -39.1258163, 35.8437691, -39.1807365, 35.9033508, -75.0291672, 75.0245056
10: -77.2260742, 47.7212410, -77.3171692, 47.8066826, -125.0327606, 125.0384064
11: -92.9816284, 14.2313957, -93.1608276, 14.3110218, -106.7332001, 106.8032227
12: -58.8658867, 51.9315338, -58.9817581, 52.0475883, -110.9134750, 110.9132919
13: -64.1183701, 66.7683258, -64.1683884, 66.8546143, -130.9729919, 130.9367065
14: -142.3578949, 20.9416828, -142.5999451, 21.0989494, -163.4568481, 163.5416260
15: -55.9344368, 46.6895332, -56.0377922, 46.7592278, -102.6936646, 102.7273254
16: -85.5276489, 29.5867386, -85.6287842, 29.6776028, -115.2052536, 115.2078094
17: -157.6787872, 22.7648430, -157.8186493, 22.8865662, -170.1296234, 170.0794067
18: -75.0799866, 48.5977325, -75.2312164, 48.7534752, -123.8334656, 123.8289490
19: -60.3920822, 14.7219152, -60.4497910, 14.7658405, -75.1579208, 75.1717072
20: -51.6937790, 18.7260647, -51.7760315, 18.7792168, -70.4729919, 70.5020981
21: -78.5317917, 18.6328545, -78.6148300, 18.6828175, -97.2146072, 97.2476807
22: -82.7314148, 24.7807159, -82.8359222, 24.8894806, -107.6208954, 107.6166382
23: -51.9689713, 24.5617599, -52.0277367, 24.5961094, -76.5650787, 76.5894928
24: -50.1737747, 33.0148849, -50.2716370, 33.0882492, -83.2620239, 83.2865219
25: -46.3452454, 35.6632195, -46.4779053, 35.8185196, -82.1637650, 82.1411285
26: -76.9218979, 47.6517982, -77.1264801, 47.7599030, -124.6818008, 124.7782745
27: -73.1140900, 30.4339638, -73.1812286, 30.4689808, -103.5830688, 103.6151886
28: -58.0445290, 31.2149391, -58.1208382, 31.2598076, -89.3043365, 89.3357773
29: -91.6434479, 15.8860188, -91.7364349, 15.9478416, -107.5912933, 107.6224518
30: -69.8655548, 37.1955566, -69.9632111, 37.2787704, -107.1443253, 107.1587677
31: -66.1632385, 18.6622314, -66.2658081, 18.7358074, -84.8990479, 84.9280396
32: -67.4447632, 42.2245445, -67.5117798, 42.2676086, -109.7123718, 109.7363281
33: -58.7477188, 82.2953339, -58.8979263, 82.4702148, -140.1258087, 140.1212158
34: -60.4687729, 68.9983368, -60.5941696, 69.1226807, -129.3061676, 129.3266602
35: -50.7373390, 76.7162170, -50.8113861, 76.7859650, -127.5233002, 127.5276031
36: -60.5996056, 67.1382904, -60.7352905, 67.1956635, -127.7952728, 127.8735809
37: -56.3634644, 61.1411667, -56.5676804, 61.2757416, -116.9188385, 116.9978714
38: -75.4035110, 82.5364304, -75.5125580, 82.5782928, -157.9818115, 158.0489807
39: -64.0875854, 87.8962097, -64.1742935, 87.9873428, -151.4145660, 151.4408875
40: -59.3607330, 57.8673897, -59.4758949, 57.9312782, -117.2920074, 117.3432846
41: -50.8831215, 44.4067078, -50.9737091, 44.4321442, -95.3152618, 95.3804169
42: -56.1530457, 40.2920837, -56.2199631, 40.3402519, -96.4933014, 96.5120468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3390733, upper bound: 92.1573289
time: 133.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3390733, upper bound: 92.2397352
time: 174.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -92.8168945, 36.9064636, -92.5766144, 36.7272797, -129.5441742, 129.4830780
1: -59.5403709, 29.8971672, -59.3741531, 29.7235413, -89.2639160, 89.2713165
2: -49.5681419, 30.8800659, -49.3999863, 30.7474213, -80.3155670, 80.2800522
3: -62.4353523, 31.2812347, -62.2667351, 31.1304169, -93.5657654, 93.5479736
4: -55.0073509, 43.6801758, -54.8909836, 43.5707092, -98.5780640, 98.5711594
5: -67.3706284, 36.9514656, -67.2087097, 36.8288574, -104.1994858, 104.1601715
6: -69.0903168, 41.6073341, -69.0129242, 41.5568008, -110.6471176, 110.6202545
7: -86.7659836, 27.2570705, -86.5544739, 27.0560989, -113.8220825, 113.8115463
8: -64.9211044, 51.9135361, -64.6697540, 51.6969833, -116.6180878, 116.5832901
9: -39.1908340, 35.9055328, -39.1258163, 35.8437691, -75.0346069, 75.0313492
10: -77.3383026, 47.8122673, -77.2260742, 47.7212410, -125.0595398, 125.0383453
11: -93.1560059, 14.3186169, -92.9816284, 14.2313957, -106.8281555, 106.7415314
12: -58.9852676, 52.0732651, -58.8658867, 51.9315338, -110.9168015, 110.9391479
13: -64.1752014, 66.8678436, -64.1183701, 66.7683258, -130.9435272, 130.9862061
14: -142.6594696, 21.1012630, -142.3578949, 20.9416828, -163.6011505, 163.4591522
15: -56.0562820, 46.7623138, -55.9344368, 46.6895332, -102.7458191, 102.6967468
16: -85.6339722, 29.6846275, -85.5276489, 29.5867386, -115.2207108, 115.2122803
17: -157.8517761, 22.8917389, -157.6787872, 22.7648430, -170.1839447, 170.1354218
18: -75.2397614, 48.7835312, -75.0799866, 48.5977325, -123.8374939, 123.8635178
19: -60.4546967, 14.7685261, -60.3920822, 14.7219152, -75.1766129, 75.1606064
20: -51.7836494, 18.7849102, -51.6937790, 18.7260647, -70.5097122, 70.4786911
21: -78.6214066, 18.6872406, -78.5317917, 18.6328545, -97.2542572, 97.2190323
22: -82.8503723, 24.8939972, -82.7314148, 24.7807159, -107.6310883, 107.6254120
23: -52.0333862, 24.6008911, -51.9689713, 24.5617599, -76.5951462, 76.5698624
24: -50.2813492, 33.0999718, -50.1737747, 33.0148849, -83.2962341, 83.2737427
25: -46.4842339, 35.8317833, -46.3452454, 35.6632195, -82.1474533, 82.1770325
26: -77.1396332, 47.7798233, -76.9218979, 47.6517982, -124.7914276, 124.7017212
27: -73.1921539, 30.4729614, -73.1140900, 30.4339638, -103.6261139, 103.5870514
28: -58.1272621, 31.2648582, -58.0445290, 31.2149391, -89.3422012, 89.3093872
29: -91.7566528, 15.9509907, -91.6434479, 15.8860188, -107.6426697, 107.5944366
30: -69.9721069, 37.2895775, -69.8655548, 37.1955566, -107.1676636, 107.1551361
31: -66.2725830, 18.7404671, -66.1632385, 18.6622314, -84.9348145, 84.9037018
32: -67.5052643, 42.2814713, -67.4447632, 42.2245445, -109.7298126, 109.7262344
33: -58.9039345, 82.5077591, -58.7477188, 82.2953339, -140.1275940, 140.1834869
34: -60.6011429, 69.1558380, -60.4687729, 68.9983368, -129.3330841, 129.3655396
35: -50.8169403, 76.8069000, -50.7373390, 76.7162170, -127.5331573, 127.5442352
36: -60.7409821, 67.2212677, -60.5996056, 67.1382904, -127.8792725, 127.8208771
37: -56.5746498, 61.3061066, -56.3634644, 61.1411667, -117.0045013, 116.9611664
38: -75.5191956, 82.5988159, -75.4035110, 82.5364304, -158.0556335, 158.0023193
39: -64.1811829, 88.0098114, -64.0875854, 87.8962097, -151.4479370, 151.4559631
40: -59.4822884, 57.9480896, -59.3607330, 57.8673897, -117.3496780, 117.3088226
41: -50.9663925, 44.4438477, -50.8831215, 44.4067078, -95.3731003, 95.3269653
42: -56.2136574, 40.3499680, -56.1530457, 40.2920837, -96.5057373, 96.5030136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2440007, upper bound: 92.2568641
time: 141.83 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2397352, upper bound: 92.3390728
time: 191.08 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -92.8168945, 36.9064636, -92.8168945, 36.9064636, -129.7233582, 129.7233582
1: -59.5403709, 29.8971672, -59.5403709, 29.8971672, -89.4375381, 89.4375381
2: -49.5681419, 30.8800659, -49.5681419, 30.8800659, -80.4482117, 80.4482117
3: -62.4353523, 31.2812347, -62.4353523, 31.2812347, -93.7165833, 93.7165833
4: -55.0073509, 43.6801758, -55.0073509, 43.6801758, -98.6875305, 98.6875305
5: -67.3706284, 36.9514656, -67.3706284, 36.9514656, -104.3220978, 104.3220978
6: -69.0903168, 41.6073341, -69.0903168, 41.6073341, -110.6976471, 110.6976471
7: -86.7659836, 27.2570705, -86.7659836, 27.2570705, -114.0230560, 114.0230560
8: -64.9211044, 51.9135361, -64.9211044, 51.9135361, -116.8346405, 116.8346405
9: -39.1908340, 35.9055328, -39.1908340, 35.9055328, -75.0963669, 75.0963669
10: -77.3383026, 47.8122673, -77.3383026, 47.8122673, -125.1505737, 125.1505737
11: -93.1560059, 14.3186169, -93.1560059, 14.3186169, -106.9089355, 106.9089203
12: -58.9852676, 52.0732651, -58.9852676, 52.0732651, -111.0585327, 111.0585327
13: -64.1752014, 66.8678436, -64.1752014, 66.8678436, -131.0430450, 131.0430450
14: -142.6594696, 21.1012630, -142.6594696, 21.1012630, -163.7607269, 163.7607269
15: -56.0562820, 46.7623138, -56.0562820, 46.7623138, -102.8185959, 102.8185959
16: -85.6339722, 29.6846275, -85.6339722, 29.6846275, -115.3186035, 115.3186035
17: -157.8517761, 22.8917389, -157.8517761, 22.8917389, -170.2267761, 170.2268066
18: -75.2397614, 48.7835312, -75.2397614, 48.7835312, -124.0232925, 124.0232925
19: -60.4546967, 14.7685261, -60.4546967, 14.7685261, -75.2232208, 75.2232208
20: -51.7836494, 18.7849102, -51.7836494, 18.7849102, -70.5685577, 70.5685577
21: -78.6214066, 18.6872406, -78.6214066, 18.6872406, -97.3086472, 97.3086472
22: -82.8503723, 24.8939972, -82.8503723, 24.8939972, -107.7443695, 107.7443695
23: -52.0333862, 24.6008911, -52.0333862, 24.6008911, -76.6342773, 76.6342773
24: -50.2813492, 33.0999718, -50.2813492, 33.0999718, -83.3813171, 83.3813171
25: -46.4842339, 35.8317833, -46.4842339, 35.8317833, -82.3160172, 82.3160172
26: -77.1396332, 47.7798233, -77.1396332, 47.7798233, -124.9194565, 124.9194565
27: -73.1921539, 30.4729614, -73.1921539, 30.4729614, -103.6651154, 103.6651154
28: -58.1272621, 31.2648582, -58.1272621, 31.2648582, -89.3921204, 89.3921204
29: -91.7566528, 15.9509907, -91.7566528, 15.9509907, -107.7076416, 107.7076416
30: -69.9721069, 37.2895775, -69.9721069, 37.2895775, -107.2616882, 107.2616882
31: -66.2725830, 18.7404671, -66.2725830, 18.7404671, -85.0130463, 85.0130463
32: -67.5052643, 42.2814713, -67.5052643, 42.2814713, -109.7867355, 109.7867355
33: -58.9039345, 82.5077591, -58.9039345, 82.5077591, -140.3224487, 140.3224487
34: -60.6011429, 69.1558380, -60.6011429, 69.1558380, -129.4639587, 129.4639435
35: -50.8169403, 76.8069000, -50.8169403, 76.8069000, -127.6238403, 127.6238403
36: -60.7409821, 67.2212677, -60.7409821, 67.2212677, -127.9622498, 127.9622498
37: -56.5746498, 61.3061066, -56.5746498, 61.3061066, -117.1603546, 117.1603470
38: -75.5191956, 82.5988159, -75.5191956, 82.5988159, -158.1180115, 158.1180115
39: -64.1811829, 88.0098114, -64.1811829, 88.0098114, -151.5265045, 151.5265045
40: -59.4822884, 57.9480896, -59.4822884, 57.9480896, -117.4303741, 117.4303741
41: -50.9663925, 44.4438477, -50.9663925, 44.4438477, -95.4102402, 95.4102402
42: -56.2136574, 40.3499680, -56.2136574, 40.3499680, -96.5636292, 96.5636292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2440007, upper bound: 92.2568645
time: 113.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2397352, upper bound: 92.3390732
time: 160.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 276.94 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2388283, upper bound: 92.1642379
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2201642, upper bound: 92.2073375
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.3390733, upper bound: 92.1573289
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.3390733, upper bound: 92.2397352
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2440007, upper bound: 92.2568641
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2397352, upper bound: 92.3390728
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2440007, upper bound: 92.2568645
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 276.94
Output dim: 37, lower bound: -92.2397352, upper bound: 92.3390732

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -92.4202118, 36.6817245, -92.5033188, 36.7625885, -129.1828003, 129.1850433
1: -59.2744102, 29.6923962, -59.3358116, 29.7628670, -89.0372772, 89.0282059
2: -49.3360291, 30.7054100, -49.4107323, 30.7721291, -80.1081543, 80.1161423
3: -62.2335815, 31.0667381, -62.3454361, 31.1619797, -93.3955612, 93.4121704
4: -54.7773819, 43.5166893, -54.8024864, 43.5294800, -98.3068619, 98.3191757
5: -67.1613770, 36.7612839, -67.2396240, 36.8255959, -103.9869690, 104.0009079
6: -68.9753265, 41.3595085, -68.9759369, 41.3756561, -110.3509827, 110.3354492
7: -86.4628906, 27.0098991, -86.5726929, 27.1177635, -113.5806580, 113.5825958
8: -64.5241470, 51.6350861, -64.6450043, 51.7505150, -116.2746582, 116.2800903
9: -39.0695724, 35.8085976, -39.0777817, 35.7813492, -74.8509216, 74.8863831
10: -77.0101929, 47.6595001, -77.0503998, 47.6062317, -124.6164246, 124.7098999
11: -92.8627319, 14.2013741, -93.0180511, 14.1987686, -106.4851685, 106.6531525
12: -58.7878075, 51.8803787, -58.8671913, 51.9765053, -110.7643127, 110.7475739
13: -64.0676575, 66.5264893, -63.9769592, 66.5916138, -130.6592712, 130.5034485
14: -141.9619141, 20.9177380, -142.1710052, 20.9198875, -162.8818054, 163.0887451
15: -55.7955246, 46.6429062, -55.8411560, 46.5995827, -102.3951111, 102.4840622
16: -85.3789520, 29.5299416, -85.4154510, 29.5114117, -114.8903656, 114.9453888
17: -157.3392029, 22.7043495, -157.4112549, 22.6529160, -169.5402069, 169.6605530
18: -74.8733978, 48.5643082, -75.0413361, 48.6285019, -123.5018997, 123.6056442
19: -60.3033066, 14.6933041, -60.3391190, 14.6989880, -75.0022964, 75.0324249
20: -51.6259155, 18.7022533, -51.6768112, 18.7083492, -70.3342667, 70.3790665
21: -78.3830261, 18.6106377, -78.4666519, 18.5836849, -96.9667130, 97.0772858
22: -82.4967651, 24.7530746, -82.6260452, 24.7935905, -107.2903595, 107.3791199
23: -51.8867416, 24.5374794, -51.9276657, 24.5341568, -76.4208984, 76.4651489
24: -49.9929848, 32.9975624, -50.1078186, 33.0318298, -83.0248108, 83.1053772
25: -46.2620735, 35.6423264, -46.3298454, 35.7616348, -82.0237122, 81.9721680
26: -76.7086563, 47.6147537, -76.9420624, 47.6296082, -124.3382645, 124.5568161
27: -72.9339371, 30.4089775, -73.0483551, 30.3819008, -103.3158417, 103.4573364
28: -57.9661827, 31.1847229, -58.0356636, 31.1985798, -89.1647644, 89.2203827
29: -91.4025574, 15.8589287, -91.5002594, 15.8313332, -107.2338867, 107.3591919
30: -69.6957092, 37.1639252, -69.8293915, 37.1925049, -106.8882141, 106.9933167
31: -66.0960999, 18.6338654, -66.1554260, 18.6564598, -84.7525635, 84.7892914
32: -67.3642731, 42.0468178, -67.3514481, 42.0348091, -109.3990784, 109.3982697
33: -58.6759720, 82.0697479, -58.6404037, 82.1865540, -139.7820435, 139.6321411
34: -60.4072952, 68.8641434, -60.4271927, 68.9154129, -129.0520935, 129.0175476
35: -50.6816139, 76.5169678, -50.5990906, 76.5230408, -127.2046509, 127.1160583
36: -60.5484390, 66.8659668, -60.5218773, 66.8545532, -127.4029922, 127.3878479
37: -56.2636375, 61.0419464, -56.3121910, 61.1454430, -116.6960297, 116.6393585
38: -75.3343277, 82.2243500, -75.2353973, 82.1971283, -157.5314636, 157.4597473
39: -63.9894028, 87.6000824, -63.8659363, 87.6198578, -150.9720001, 150.8286438
40: -59.2701454, 57.7009163, -59.3067970, 57.7178421, -116.9879913, 117.0077133
41: -50.8411865, 44.2520752, -50.8396912, 44.2585182, -95.0997009, 95.0917664
42: -56.1175461, 40.1565781, -56.1260338, 40.1927338, -96.3102798, 96.2826080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
time: 88.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
time: 125.44 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -92.6764450, 36.8072090, -92.5124588, 36.7661057, -129.4425507, 129.3196716
1: -59.3982353, 29.8779449, -59.3423309, 29.7652016, -89.1634369, 89.2202759
2: -49.4048042, 30.8033524, -49.4125061, 30.7755165, -80.1803207, 80.2158585
3: -62.3056946, 31.2083549, -62.3471336, 31.1672878, -93.4729843, 93.5554886
4: -54.8823395, 43.7161179, -54.8086929, 43.5337601, -98.4160995, 98.5248108
5: -67.2279205, 36.8911247, -67.2416992, 36.8313828, -104.0593033, 104.1328278
6: -69.2095184, 41.5628052, -68.9784088, 41.3918228, -110.6013412, 110.5412140
7: -86.5353165, 27.1765461, -86.5762482, 27.1217232, -113.6570435, 113.7527924
8: -64.6426239, 51.8435173, -64.6531830, 51.7554665, -116.3980865, 116.4967041
9: -39.2183456, 35.9720383, -39.0818634, 35.7839050, -75.0022507, 75.0539017
10: -77.2067947, 47.9548416, -77.0674973, 47.6109581, -124.8177490, 125.0223389
11: -93.0154114, 14.4224911, -93.0274963, 14.2010040, -106.6588745, 106.8900299
12: -58.9173088, 51.9707222, -58.8738441, 51.9791718, -110.8964844, 110.8445663
13: -64.4521637, 66.7653732, -63.9806061, 66.6108704, -131.0630341, 130.7459717
14: -142.3637085, 21.2319851, -142.2035828, 20.9214039, -163.2851105, 163.4355621
15: -55.9315338, 46.8813782, -55.8519707, 46.6031761, -102.5347137, 102.7333527
16: -85.5863342, 29.7721558, -85.4266205, 29.5161438, -115.1024780, 115.1987762
17: -157.6469727, 23.1090126, -157.4387207, 22.6578560, -169.8673248, 170.0967407
18: -75.0882263, 48.8372078, -75.0589447, 48.6307411, -123.7189636, 123.8961487
19: -60.4338455, 14.7976742, -60.3463097, 14.7012653, -75.1351089, 75.1439819
20: -51.7357597, 18.8579273, -51.6821060, 18.7100163, -70.4457779, 70.5400314
21: -78.5671387, 18.7891502, -78.4795303, 18.5850372, -97.1521759, 97.2686768
22: -82.7382050, 24.9342880, -82.6468964, 24.7954903, -107.5336914, 107.5811844
23: -52.0228233, 24.6625252, -51.9342041, 24.5357075, -76.5585327, 76.5967255
24: -50.3133240, 33.1511879, -50.1240158, 33.0330811, -83.3464050, 83.2752075
25: -46.4775848, 35.7602272, -46.3364944, 35.7629776, -82.2405624, 82.0967255
26: -76.9686127, 47.9258041, -76.9610748, 47.6321030, -124.6007156, 124.8868790
27: -73.1492462, 30.6188698, -73.0642166, 30.3835392, -103.5327835, 103.6830902
28: -58.0811462, 31.3016968, -58.0417671, 31.2006664, -89.2818146, 89.3434601
29: -91.6596527, 16.0655861, -91.5211258, 15.8332806, -107.4929352, 107.5867157
30: -69.9272003, 37.3736687, -69.8437653, 37.1947403, -107.1219406, 107.2174377
31: -66.2467651, 18.7591209, -66.1606140, 18.6585064, -84.9052734, 84.9197388
32: -67.6339722, 42.2655449, -67.3581924, 42.0495911, -109.6835632, 109.6237335
33: -59.1073303, 82.2481232, -58.6456184, 82.2040863, -140.2321014, 139.8187561
34: -60.6542587, 68.9909210, -60.4315681, 68.9250793, -129.3112488, 129.1534424
35: -51.0411835, 76.6721039, -50.6031418, 76.5381241, -127.5793076, 127.2752457
36: -60.8825722, 67.0714417, -60.5259819, 66.8755188, -127.7580872, 127.5974274
37: -56.6852798, 61.1215706, -56.3200569, 61.1528015, -117.1291580, 116.7280045
38: -75.7732849, 82.4884872, -75.2407990, 82.2211304, -157.9944153, 157.7292786
39: -64.5231247, 87.8171768, -63.8737564, 87.6424332, -151.5325317, 151.0568695
40: -59.5465164, 57.8595734, -59.3139229, 57.7310600, -117.2775726, 117.1734924
41: -51.0910606, 44.4114952, -50.8426628, 44.2702522, -95.3613129, 95.2541580
42: -56.2774467, 40.3160324, -56.1286278, 40.2039070, -96.4813538, 96.4446564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
time: 71.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
time: 99.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -92.3136139, 36.5905609, -92.7731857, 36.9024582, -129.2160645, 129.3637390
1: -59.2036133, 29.5968208, -59.5093803, 29.8914948, -89.0951080, 89.1062012
2: -49.2729111, 30.6456985, -49.5404282, 30.8767815, -80.1496887, 80.1861267
3: -62.2051926, 31.0205917, -62.4032021, 31.2763634, -93.4815521, 93.4237976
4: -54.7121887, 43.4280968, -54.9850502, 43.6755486, -98.3877411, 98.4131470
5: -67.1045074, 36.7111931, -67.3426208, 36.9483109, -104.0528183, 104.0538177
6: -68.8884354, 41.3498650, -69.1026154, 41.5941887, -110.4826202, 110.4524841
7: -86.3938446, 26.9258995, -86.7364960, 27.2512074, -113.6450500, 113.6623993
8: -64.4474335, 51.5426826, -64.8718414, 51.9088860, -116.3563232, 116.4145203
9: -39.0250511, 35.7236633, -39.1807365, 35.9033508, -74.9284058, 74.9044037
10: -76.9707565, 47.5243340, -77.3171692, 47.8066826, -124.7774353, 124.8415070
11: -92.8457031, 14.1207809, -93.1608276, 14.3110218, -106.5915985, 106.7183685
12: -58.7570419, 51.8586197, -58.9817581, 52.0475883, -110.8046265, 110.8403778
13: -63.9292984, 66.5201035, -64.1683884, 66.8546143, -130.7839050, 130.6884918
14: -141.9527435, 20.7636433, -142.5999451, 21.0989494, -163.0516968, 163.3635864
15: -55.7440834, 46.5329399, -56.0377922, 46.7592278, -102.5033112, 102.5707321
16: -85.3208466, 29.4242210, -85.6287842, 29.6776028, -114.9984512, 115.0530090
17: -157.2899323, 22.5353851, -157.8186493, 22.8865662, -169.7242279, 169.9182739
18: -74.9051056, 48.4742432, -75.2312164, 48.7534752, -123.6585846, 123.7054596
19: -60.2866096, 14.6571159, -60.4497910, 14.7658405, -75.0524521, 75.1069031
20: -51.5993767, 18.6562786, -51.7760315, 18.7792168, -70.3785934, 70.4323120
21: -78.3938751, 18.5347767, -78.6148300, 18.6828175, -97.0766907, 97.1496048
22: -82.5383453, 24.6864796, -82.8359222, 24.8894806, -107.4278259, 107.5223999
23: -51.8742943, 24.5009155, -52.0277367, 24.5961094, -76.4704056, 76.5286560
24: -50.0239906, 32.9593582, -50.2716370, 33.0882492, -83.1122437, 83.2309952
25: -46.2029533, 35.6072159, -46.4779053, 35.8185196, -82.0214691, 82.0851212
26: -76.7549591, 47.5233841, -77.1264801, 47.7599030, -124.5148621, 124.6498642
27: -72.9947052, 30.3480320, -73.1812286, 30.4689808, -103.4636841, 103.5292587
28: -57.9645309, 31.1550789, -58.1208382, 31.2598076, -89.2243347, 89.2759171
29: -91.4226532, 15.7712669, -91.7364349, 15.9478416, -107.3704987, 107.5077057
30: -69.7454376, 37.1109428, -69.9632111, 37.2787704, -107.0242081, 107.0741577
31: -66.0572739, 18.5845509, -66.2658081, 18.7358074, -84.7930832, 84.8503571
32: -67.2898712, 42.0011368, -67.5117798, 42.2676086, -109.5574799, 109.5129166
33: -58.4940872, 82.0222321, -58.8979263, 82.4702148, -139.8874207, 139.8460083
34: -60.3054276, 68.7951355, -60.5941696, 69.1226807, -129.1663208, 129.1178589
35: -50.5279999, 76.4622040, -50.8113861, 76.7859650, -127.3139648, 127.2735901
36: -60.3890152, 66.8098526, -60.7352905, 67.1956635, -127.5846786, 127.5451431
37: -56.1139832, 61.0148468, -56.5676804, 61.2757416, -116.6775742, 116.8705444
38: -75.1305008, 82.1700287, -75.5125580, 82.5782928, -157.7088013, 157.6825867
39: -63.7851028, 87.5419006, -64.1742935, 87.9873428, -151.1403046, 151.0832825
40: -59.1976013, 57.6620064, -59.4758949, 57.9312782, -117.1288757, 117.1379013
41: -50.7513390, 44.2424469, -50.9737091, 44.4321442, -95.1834869, 95.2161560
42: -56.0612755, 40.1529388, -56.2199631, 40.3402519, -96.4015274, 96.3729019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1768818, upper bound: 92.1393062
time: 140.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2201642, upper bound: 92.1207193
time: 140.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -92.5695038, 36.7256088, -92.7731857, 36.9024582, -129.4719543, 129.4987946
1: -59.3686676, 29.7222290, -59.5093803, 29.8914948, -89.2601624, 89.2316132
2: -49.3975220, 30.7459259, -49.5404282, 30.8767815, -80.2743073, 80.2863541
3: -62.2601357, 31.1281586, -62.4032021, 31.2763634, -93.5364990, 93.5313568
4: -54.8846397, 43.5684624, -54.9850502, 43.6755486, -98.5601883, 98.5535126
5: -67.2040558, 36.8267288, -67.3426208, 36.9483109, -104.1523666, 104.1693497
6: -69.0111694, 41.5491295, -69.1026154, 41.5941887, -110.6053619, 110.6517487
7: -86.5500946, 27.0543671, -86.7364960, 27.2512074, -113.8013000, 113.7908630
8: -64.6617661, 51.6946106, -64.8718414, 51.9088860, -116.5706482, 116.5664520
9: -39.1227798, 35.8420563, -39.1807365, 35.9033508, -75.0261307, 75.0227966
10: -77.2173767, 47.7183800, -77.3171692, 47.8066826, -125.0240631, 125.0355530
11: -92.9767303, 14.2299519, -93.1608276, 14.3110218, -106.6975632, 106.8016815
12: -58.8634491, 51.9249840, -58.9817581, 52.0475883, -110.9110413, 110.9067383
13: -64.1156464, 66.7593079, -64.1683884, 66.8546143, -130.9702606, 130.9277039
14: -142.3428497, 20.9403877, -142.5999451, 21.0989494, -163.4418030, 163.5403290
15: -55.9280281, 46.6876526, -56.0377922, 46.7592278, -102.6872559, 102.7254486
16: -85.5204010, 29.5845795, -85.6287842, 29.6776028, -115.1926270, 115.2055664
17: -157.6652222, 22.7625427, -157.8186493, 22.8865662, -170.0476379, 170.0769958
18: -75.0736847, 48.5959663, -75.2312164, 48.7534752, -123.8271637, 123.8271790
19: -60.3883247, 14.7208900, -60.4497910, 14.7658405, -75.1541672, 75.1706848
20: -51.6913490, 18.7247658, -51.7760315, 18.7792168, -70.4705658, 70.5007935
21: -78.5263824, 18.6318035, -78.6148300, 18.6828175, -97.2091980, 97.2466354
22: -82.7238922, 24.7795563, -82.8359222, 24.8894806, -107.6133728, 107.6154785
23: -51.9657516, 24.5604820, -52.0277367, 24.5961094, -76.5618591, 76.5882187
24: -50.1676903, 33.0140190, -50.2716370, 33.0882492, -83.2559357, 83.2856598
25: -46.3420906, 35.6620560, -46.4779053, 35.8185196, -82.1606140, 82.1399612
26: -76.9165802, 47.6499405, -77.1264801, 47.7599030, -124.6764832, 124.7764206
27: -73.1078491, 30.4326401, -73.1812286, 30.4689808, -103.5768280, 103.6138687
28: -58.0413132, 31.2133579, -58.1208382, 31.2598076, -89.3011169, 89.3341980
29: -91.6343842, 15.8849487, -91.7364349, 15.9478416, -107.5822296, 107.6213837
30: -69.8604126, 37.1940765, -69.9632111, 37.2787704, -107.1391830, 107.1572876
31: -66.1605682, 18.6610165, -66.2658081, 18.7358074, -84.8963776, 84.9268265
32: -67.4416046, 42.2165298, -67.5117798, 42.2676086, -109.7092133, 109.7283096
33: -58.7444916, 82.2852097, -58.8979263, 82.4702148, -140.1225739, 140.0954285
34: -60.4660187, 68.9908752, -60.5941696, 69.1226807, -129.3033600, 129.2937927
35: -50.7345238, 76.7072296, -50.8113861, 76.7859650, -127.5204926, 127.5186157
36: -60.5971680, 67.1262054, -60.7352905, 67.1956635, -127.7928314, 127.8614960
37: -56.3591576, 61.1361694, -56.5676804, 61.2757416, -116.9144897, 116.9831161
38: -75.4006653, 82.5223236, -75.5125580, 82.5782928, -157.9789581, 158.0348816
39: -64.0832062, 87.8824692, -64.1742935, 87.9873428, -151.4101868, 151.3988342
40: -59.3571739, 57.8600349, -59.4758949, 57.9312782, -117.2884521, 117.3359299
41: -50.8811607, 44.4004860, -50.9737091, 44.4321442, -95.3133087, 95.3741913
42: -56.1516380, 40.2865906, -56.2199631, 40.3402519, -96.4918900, 96.5065536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1768818, upper bound: 92.2215954
time: 111.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2201642, upper bound: 92.2039411
time: 99.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -92.5540314, 36.7697220, -92.5336609, 36.7205124, -129.2745361, 129.3033752
1: -59.3698997, 29.7704430, -59.3439713, 29.7184563, -89.0883560, 89.1144104
2: -49.4411049, 30.7782860, -49.3767395, 30.7412491, -80.1823578, 80.1550293
3: -62.3738174, 31.1712894, -62.2566872, 31.1210518, -93.4948730, 93.4279785
4: -54.8285713, 43.5375519, -54.8578033, 43.5626831, -98.3912506, 98.3953552
5: -67.2664337, 36.8337517, -67.1897278, 36.8194427, -104.0858765, 104.0234833
6: -68.9657593, 41.4004135, -69.0069275, 41.5186310, -110.4843903, 110.4073410
7: -86.6053772, 27.1268291, -86.5239639, 27.0499420, -113.6553192, 113.6507950
8: -64.6988220, 51.7591705, -64.6268921, 51.6872635, -116.3860855, 116.3860626
9: -39.0900726, 35.7853966, -39.1124420, 35.8375168, -74.9275894, 74.8978424
10: -77.0830307, 47.6153564, -77.1770782, 47.7115746, -124.7946014, 124.7924347
11: -93.0200806, 14.2080193, -92.9591293, 14.2262478, -106.6811371, 106.6048813
12: -58.8763657, 52.0004234, -58.8543129, 51.9195404, -110.7959061, 110.8547363
13: -63.9860725, 66.6195526, -64.1085510, 66.7223206, -130.7083893, 130.7281036
14: -142.2544556, 20.9232330, -142.2827301, 20.9364548, -163.1909180, 163.2059631
15: -55.8659325, 46.6056442, -55.8991776, 46.6820374, -102.5479736, 102.5048218
16: -85.4272308, 29.5221214, -85.4919968, 29.5779305, -115.0051575, 115.0141144
17: -157.4630432, 22.6622009, -157.6047668, 22.7567253, -169.7705994, 169.8304901
18: -75.0649261, 48.6600342, -75.0485992, 48.5908623, -123.6557922, 123.7086334
19: -60.3492737, 14.7037401, -60.3769684, 14.7183037, -75.0675812, 75.0807114
20: -51.6892319, 18.7151184, -51.6842537, 18.7213516, -70.4105835, 70.3993683
21: -78.4835281, 18.5891762, -78.5101929, 18.6279831, -97.1115112, 97.0993652
22: -82.6573029, 24.7997799, -82.6970062, 24.7757206, -107.4330215, 107.4967880
23: -51.9386749, 24.5400314, -51.9567184, 24.5567646, -76.4954376, 76.4967499
24: -50.1314888, 33.0444412, -50.1570587, 33.0118408, -83.1433258, 83.2014999
25: -46.3418274, 35.7757683, -46.3334351, 35.6588669, -82.0006943, 82.1092072
26: -76.9726868, 47.6514587, -76.8965073, 47.6451569, -124.6178436, 124.5479660
27: -73.0727539, 30.3870010, -73.0950775, 30.4288445, -103.5016022, 103.4820786
28: -58.0472374, 31.2050018, -58.0337715, 31.2086010, -89.2558365, 89.2387695
29: -91.5359039, 15.8361874, -91.6035461, 15.8816967, -107.4176025, 107.4397354
30: -69.8519287, 37.2049522, -69.8485565, 37.1894073, -107.0413361, 107.0535126
31: -66.1665192, 18.6627769, -66.1528320, 18.6572609, -84.8237762, 84.8156128
32: -67.3503265, 42.0581779, -67.4342041, 42.1851273, -109.5354538, 109.4923859
33: -58.6503410, 82.2347031, -58.7343979, 82.2406235, -139.8185730, 139.8948669
34: -60.4378052, 68.9526367, -60.4585800, 68.9591522, -129.1279755, 129.1461182
35: -50.6075287, 76.5529327, -50.7278290, 76.6649780, -127.2725067, 127.2807617
36: -60.5304298, 66.8928528, -60.5911674, 67.0713806, -127.6018066, 127.4840240
37: -56.3251114, 61.1798515, -56.3473282, 61.1161537, -116.7282486, 116.8177032
38: -75.2460938, 82.2324219, -75.3908844, 82.4629440, -157.7090454, 157.6233063
39: -63.8785286, 87.6555634, -64.0718536, 87.8229523, -151.0714417, 151.0827789
40: -59.3191528, 57.7427635, -59.3475113, 57.8283386, -117.1474915, 117.0902710
41: -50.8346329, 44.2795219, -50.8754883, 44.3766212, -95.2112579, 95.1550140
42: -56.1218376, 40.2108231, -56.1466141, 40.2682800, -96.3901215, 96.3574371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388280
time: 93.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201639
time: 96.52 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -92.8097992, 36.9047928, -92.5766144, 36.7272797, -129.5370789, 129.4814148
1: -59.5348930, 29.8958378, -59.3741531, 29.7235413, -89.2584381, 89.2699890
2: -49.5656815, 30.8785629, -49.3999863, 30.7474213, -80.3131027, 80.2785492
3: -62.4287720, 31.2790031, -62.2667351, 31.1304169, -93.5591888, 93.5457382
4: -55.0009804, 43.6779099, -54.8909836, 43.5707092, -98.5716858, 98.5688934
5: -67.3659668, 36.9493752, -67.2087097, 36.8288574, -104.1948242, 104.1580811
6: -69.0885391, 41.5996895, -69.0129242, 41.5568008, -110.6453400, 110.6126099
7: -86.7616043, 27.2553329, -86.5544739, 27.0560989, -113.8177032, 113.8098068
8: -64.9131165, 51.9111557, -64.6697540, 51.6969833, -116.6100998, 116.5809097
9: -39.1877975, 35.9038086, -39.1258163, 35.8437691, -75.0315704, 75.0296249
10: -77.3296051, 47.8094292, -77.2260742, 47.7212410, -125.0508423, 125.0355072
11: -93.1510925, 14.3172092, -92.9816284, 14.2313957, -106.7925415, 106.7399979
12: -58.9828300, 52.0667076, -58.8658867, 51.9315338, -110.9143677, 110.9325943
13: -64.1724930, 66.8588104, -64.1183701, 66.7683258, -130.9408264, 130.9771729
14: -142.6444092, 21.1000099, -142.3578949, 20.9416828, -163.5860901, 163.4579010
15: -56.0498466, 46.7604256, -55.9344368, 46.6895332, -102.7393799, 102.6948624
16: -85.6267395, 29.6824493, -85.5276489, 29.5867386, -115.1993256, 115.2100983
17: -157.8382263, 22.8894005, -157.6787872, 22.7648430, -170.1019745, 170.1330261
18: -75.2334747, 48.7817345, -75.0799866, 48.5977325, -123.8312073, 123.8617249
19: -60.4509583, 14.7674961, -60.3920822, 14.7219152, -75.1728745, 75.1595764
20: -51.7812309, 18.7835999, -51.6937790, 18.7260647, -70.5072937, 70.4773788
21: -78.6160278, 18.6861496, -78.5317917, 18.6328545, -97.2488861, 97.2179413
22: -82.8428192, 24.8928432, -82.7314148, 24.7807159, -107.6235352, 107.6242599
23: -52.0301743, 24.5996075, -51.9689713, 24.5617599, -76.5919342, 76.5685806
24: -50.2752991, 33.0991058, -50.1737747, 33.0148849, -83.2901840, 83.2728806
25: -46.4810905, 35.8306122, -46.3452454, 35.6632195, -82.1443100, 82.1758575
26: -77.1342850, 47.7779808, -76.9218979, 47.6517982, -124.7860870, 124.6998749
27: -73.1859283, 30.4716263, -73.1140900, 30.4339638, -103.6198883, 103.5857162
28: -58.1240578, 31.2632675, -58.0445290, 31.2149391, -89.3389969, 89.3078003
29: -91.7475967, 15.9498997, -91.6434479, 15.8860188, -107.6336136, 107.5933456
30: -69.9669571, 37.2880859, -69.8655548, 37.1955566, -107.1625137, 107.1536407
31: -66.2698975, 18.7392483, -66.1632385, 18.6622314, -84.9321289, 84.9024887
32: -67.5021210, 42.2734451, -67.4447632, 42.2245445, -109.7266693, 109.7182083
33: -58.9007263, 82.4976501, -58.7477188, 82.2953339, -140.1243896, 140.1576843
34: -60.5984230, 69.1483765, -60.4687729, 68.9983368, -129.3302460, 129.3327026
35: -50.8141174, 76.7979279, -50.7373390, 76.7162170, -127.5303345, 127.5352631
36: -60.7385712, 67.2091827, -60.5996056, 67.1382904, -127.8768616, 127.8087921
37: -56.5703392, 61.3011169, -56.3634644, 61.1411667, -117.0001678, 116.9464111
38: -75.5163574, 82.5847321, -75.4035110, 82.5364304, -158.0527954, 157.9882507
39: -64.1767578, 87.9960938, -64.0875854, 87.8962097, -151.4435425, 151.4138794
40: -59.4787521, 57.9407349, -59.3607330, 57.8673897, -117.3461456, 117.3014679
41: -50.9644241, 44.4376183, -50.8831215, 44.4067078, -95.3711319, 95.3207397
42: -56.2122345, 40.3444672, -56.1530457, 40.2920837, -96.5043182, 96.4975128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
time: 94.93 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
time: 91.97 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -92.5540314, 36.7697220, -92.7739716, 36.8996964, -129.4537354, 129.5437012
1: -59.3698997, 29.7704430, -59.5101852, 29.8920555, -89.2619553, 89.2806244
2: -49.4411049, 30.7782860, -49.5449066, 30.8738785, -80.3149872, 80.3231964
3: -62.3738174, 31.1712894, -62.4253159, 31.2718887, -93.6457062, 93.5966034
4: -54.8285713, 43.5375519, -54.9741631, 43.6721497, -98.5007172, 98.5117188
5: -67.2664337, 36.8337517, -67.3516541, 36.9420547, -104.2084885, 104.1854095
6: -68.9657593, 41.4004135, -69.0843201, 41.5691681, -110.5349274, 110.4847336
7: -86.6053772, 27.1268291, -86.7354889, 27.2509155, -113.8562927, 113.8623199
8: -64.6988220, 51.7591705, -64.8782501, 51.9037857, -116.6026077, 116.6374207
9: -39.0900726, 35.7853966, -39.1774635, 35.8992767, -74.9893494, 74.9628601
10: -77.0830307, 47.6153564, -77.2893219, 47.8025780, -124.8856049, 124.9046783
11: -93.0200806, 14.2080193, -93.1334915, 14.3134975, -106.7619247, 106.7722778
12: -58.8763657, 52.0004234, -58.9737549, 52.0612984, -110.9376678, 110.9741821
13: -63.9860725, 66.6195526, -64.1653595, 66.8218384, -130.8079071, 130.7849121
14: -142.2544556, 20.9232330, -142.5843201, 21.0960445, -163.3504944, 163.5075531
15: -55.8659325, 46.6056442, -56.0210457, 46.7547836, -102.6207123, 102.6266937
16: -85.4272308, 29.5221214, -85.5983276, 29.6757851, -115.1030121, 115.1204529
17: -157.4630432, 22.6622009, -157.7778015, 22.8835449, -169.8134155, 169.9218903
18: -75.0649261, 48.6600342, -75.2084045, 48.7766342, -123.8415604, 123.8684387
19: -60.3492737, 14.7037401, -60.4395981, 14.7649136, -75.1141891, 75.1433411
20: -51.6892319, 18.7151184, -51.7741356, 18.7802048, -70.4694366, 70.4892578
21: -78.4835281, 18.5891762, -78.5998230, 18.6823463, -97.1658783, 97.1889954
22: -82.6573029, 24.7997799, -82.8159485, 24.8890114, -107.5463104, 107.6157303
23: -51.9386749, 24.5400314, -52.0211449, 24.5958939, -76.5345688, 76.5611725
24: -50.1314888, 33.0444412, -50.2646446, 33.0969162, -83.2284088, 83.3090820
25: -46.3418274, 35.7757683, -46.4724350, 35.8274193, -82.1692505, 82.2481995
26: -76.9726868, 47.6514587, -77.1142502, 47.7731895, -124.7458801, 124.7657089
27: -73.0727539, 30.3870010, -73.1731567, 30.4678383, -103.5405884, 103.5601578
28: -58.0472374, 31.2050018, -58.1165047, 31.2585163, -89.3057556, 89.3215027
29: -91.5359039, 15.8361874, -91.7167816, 15.9466457, -107.4825516, 107.5529709
30: -69.8519287, 37.2049522, -69.9551163, 37.2834244, -107.1353531, 107.1600647
31: -66.1665192, 18.6627769, -66.2621918, 18.7354851, -84.9020081, 84.9249725
32: -67.3503265, 42.0581779, -67.4947052, 42.2420502, -109.5923767, 109.5528870
33: -58.6503410, 82.2347031, -58.8906364, 82.4530640, -140.0134277, 140.0338440
34: -60.4378052, 68.9526367, -60.5909882, 69.1166458, -129.2588806, 129.2444763
35: -50.6075287, 76.5529327, -50.8074417, 76.7556610, -127.3631897, 127.3603745
36: -60.5304298, 66.8928528, -60.7325821, 67.1543579, -127.6847839, 127.6254349
37: -56.3251114, 61.1798515, -56.5585365, 61.2811050, -116.8840942, 117.0168762
38: -75.2460938, 82.2324219, -75.5065765, 82.5253372, -157.7714233, 157.7389984
39: -63.8785286, 87.6555634, -64.1654053, 87.9365921, -151.1500397, 151.1533356
40: -59.3191528, 57.7427635, -59.4690819, 57.9089928, -117.2281494, 117.2118454
41: -50.8346329, 44.2795219, -50.9587593, 44.4137917, -95.2484283, 95.2382812
42: -56.1218376, 40.2108231, -56.2072105, 40.3261642, -96.4479980, 96.4180298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388281
time: 143.99 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201642
time: 111.46 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -92.8097992, 36.9047928, -92.8168945, 36.9064636, -129.7162628, 129.7216797
1: -59.5348930, 29.8958378, -59.5403709, 29.8971672, -89.4320602, 89.4362106
2: -49.5656815, 30.8785629, -49.5681419, 30.8800659, -80.4457474, 80.4467010
3: -62.4287720, 31.2790031, -62.4353523, 31.2812347, -93.7100067, 93.7143555
4: -55.0009804, 43.6779099, -55.0073509, 43.6801758, -98.6811523, 98.6852570
5: -67.3659668, 36.9493752, -67.3706284, 36.9514656, -104.3174286, 104.3200073
6: -69.0885391, 41.5996895, -69.0903168, 41.6073341, -110.6958771, 110.6900024
7: -86.7616043, 27.2553329, -86.7659836, 27.2570705, -114.0186768, 114.0213165
8: -64.9131165, 51.9111557, -64.9211044, 51.9135361, -116.8266525, 116.8322601
9: -39.1877975, 35.9038086, -39.1908340, 35.9055328, -75.0933304, 75.0946426
10: -77.3296051, 47.8094292, -77.3383026, 47.8122673, -125.1418762, 125.1477356
11: -93.1510925, 14.3172092, -93.1560059, 14.3186169, -106.8733368, 106.9074020
12: -58.9828300, 52.0667076, -58.9852676, 52.0732651, -111.0560913, 111.0519714
13: -64.1724930, 66.8588104, -64.1752014, 66.8678436, -131.0403442, 131.0340118
14: -142.6444092, 21.1000099, -142.6594696, 21.1012630, -163.7456665, 163.7594757
15: -56.0498466, 46.7604256, -56.0562820, 46.7623138, -102.8121643, 102.8167114
16: -85.6267395, 29.6824493, -85.6339722, 29.6846275, -115.2948608, 115.3164215
17: -157.8382263, 22.8894005, -157.8517761, 22.8917389, -170.1448059, 170.2244110
18: -75.2334747, 48.7817345, -75.2397614, 48.7835312, -124.0170059, 124.0214996
19: -60.4509583, 14.7674961, -60.4546967, 14.7685261, -75.2194824, 75.2221909
20: -51.7812309, 18.7835999, -51.7836494, 18.7849102, -70.5661392, 70.5672455
21: -78.6160278, 18.6861496, -78.6214066, 18.6872406, -97.3032684, 97.3075562
22: -82.8428192, 24.8928432, -82.8503723, 24.8939972, -107.7368164, 107.7432175
23: -52.0301743, 24.5996075, -52.0333862, 24.6008911, -76.6310654, 76.6329956
24: -50.2752991, 33.0991058, -50.2813492, 33.0999718, -83.3752747, 83.3804550
25: -46.4810905, 35.8306122, -46.4842339, 35.8317833, -82.3128738, 82.3148499
26: -77.1342850, 47.7779808, -77.1396332, 47.7798233, -124.9141083, 124.9176178
27: -73.1859283, 30.4716263, -73.1921539, 30.4729614, -103.6588898, 103.6637802
28: -58.1240578, 31.2632675, -58.1272621, 31.2648582, -89.3889160, 89.3905334
29: -91.7475967, 15.9498997, -91.7566528, 15.9509907, -107.6985855, 107.7065506
30: -69.9669571, 37.2880859, -69.9721069, 37.2895775, -107.2565308, 107.2601929
31: -66.2698975, 18.7392483, -66.2725830, 18.7404671, -85.0103607, 85.0118332
32: -67.5021210, 42.2734451, -67.5052643, 42.2814713, -109.7835922, 109.7787094
33: -58.9007263, 82.4976501, -58.9039345, 82.5077591, -140.3192291, 140.2966614
34: -60.5984230, 69.1483765, -60.6011429, 69.1558380, -129.4611206, 129.4310913
35: -50.8141174, 76.7979279, -50.8169403, 76.8069000, -127.6210175, 127.6148682
36: -60.7385712, 67.2091827, -60.7409821, 67.2212677, -127.9598389, 127.9501648
37: -56.5703392, 61.3011169, -56.5746498, 61.3061066, -117.1560364, 117.1455841
38: -75.5163574, 82.5847321, -75.5191956, 82.5988159, -158.1151733, 158.1039276
39: -64.1767578, 87.9960938, -64.1811829, 88.0098114, -151.5221252, 151.4844360
40: -59.4787521, 57.9407349, -59.4822884, 57.9480896, -117.4268417, 117.4230194
41: -50.9644241, 44.4376183, -50.9663925, 44.4438477, -95.4082718, 95.4040070
42: -56.2122345, 40.3444672, -56.2136574, 40.3499680, -96.5622025, 96.5581207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731
time: 101.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731
time: 91.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 195.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1768818, upper bound: 92.1393062
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.2201642, upper bound: 92.1207193
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1768818, upper bound: 92.2215954
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.2201642, upper bound: 92.2039411
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388280
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201639
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388281
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201642
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 195.09
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -92.4202118, 36.6817245, -92.2995300, 36.5857544, -129.0059662, 128.9812622
1: -59.2744102, 29.6923962, -59.1950226, 29.5935936, -88.8680038, 88.8874207
2: -49.3360291, 30.7054100, -49.2677879, 30.6413231, -79.9773560, 79.9731979
3: -62.2335815, 31.0667381, -62.2023544, 31.0139198, -93.2474976, 93.2690887
4: -54.7773819, 43.5166893, -54.7020035, 43.4224167, -98.1997986, 98.2186890
5: -67.1613770, 36.7612839, -67.1010437, 36.7040634, -103.8654404, 103.8623276
6: -68.9753265, 41.3595085, -68.8845444, 41.3305893, -110.3059158, 110.2440491
7: -86.4628906, 27.0098991, -86.3862076, 26.9209309, -113.3838196, 113.3961029
8: -64.5241470, 51.6350861, -64.4348984, 51.5363045, -116.0604553, 116.0699844
9: -39.0695724, 35.8085976, -39.0198059, 35.7200813, -74.7896576, 74.8283997
10: -77.0101929, 47.6595001, -76.9505768, 47.5179443, -124.5281372, 124.6100769
11: -92.8627319, 14.2013741, -92.8338928, 14.1177349, -106.4065094, 106.4703369
12: -58.7878075, 51.8803787, -58.7489090, 51.8538055, -110.6416168, 110.6292877
13: -64.0676575, 66.5264893, -63.9242973, 66.4963684, -130.5640259, 130.4507904
14: -141.9619141, 20.9177380, -141.9138031, 20.7613640, -162.7232819, 162.8315430
15: -55.7955246, 46.6429062, -55.7313423, 46.5280991, -102.3236237, 102.3742523
16: -85.3789520, 29.5299416, -85.3070221, 29.4183464, -114.7973022, 114.8369598
17: -157.3392029, 22.7043495, -157.2577057, 22.5289154, -169.4112396, 169.5020752
18: -74.8733978, 48.5643082, -74.8837967, 48.4709854, -123.3443832, 123.4481049
19: -60.3033066, 14.6933041, -60.2776108, 14.6540365, -74.9573441, 74.9709167
20: -51.6259155, 18.7022533, -51.5921555, 18.6539364, -70.2798538, 70.2944107
21: -78.3830261, 18.6106377, -78.3781662, 18.5326614, -96.9156876, 96.9888000
22: -82.4967651, 24.7530746, -82.5139847, 24.6836967, -107.1804657, 107.2670593
23: -51.8867416, 24.5374794, -51.8656845, 24.4985352, -76.3852768, 76.4031677
24: -49.9929848, 32.9975624, -50.0039482, 32.9576149, -82.9505997, 83.0015106
25: -46.2620735, 35.6423264, -46.1941414, 35.6051826, -81.8672562, 81.8364716
26: -76.7086563, 47.6147537, -76.7321014, 47.5196381, -124.2282944, 124.3468552
27: -72.9339371, 30.4089775, -72.9750366, 30.3455811, -103.2795181, 103.3840179
28: -57.9661827, 31.1847229, -57.9561615, 31.1521454, -89.1183319, 89.1408844
29: -91.4025574, 15.8589287, -91.3981934, 15.7684669, -107.1710205, 107.2571259
30: -69.6957092, 37.1639252, -69.7266388, 37.1078110, -106.8035202, 106.8905640
31: -66.0960999, 18.6338654, -66.0502777, 18.5816650, -84.6777649, 84.6841431
32: -67.3642731, 42.0468178, -67.2813187, 41.9835510, -109.3478241, 109.3281403
33: -58.6759720, 82.0697479, -58.4869270, 82.0015182, -139.5950012, 139.4722595
34: -60.4072952, 68.8641434, -60.2991028, 68.7835999, -128.9089966, 128.8840027
35: -50.6816139, 76.5169678, -50.5222931, 76.4442749, -127.1258850, 127.0392609
36: -60.5484390, 66.8659668, -60.3837776, 66.7850800, -127.3335190, 127.2497406
37: -56.2636375, 61.0419464, -56.1037216, 61.0058365, -116.5523911, 116.4272461
38: -75.3343277, 82.2243500, -75.1235580, 82.1412048, -157.4755249, 157.3479004
39: -63.9894028, 87.6000824, -63.7750168, 87.5149689, -150.8750458, 150.7431946
40: -59.2701454, 57.7009163, -59.1880951, 57.6465569, -116.9167023, 116.8890076
41: -50.8411865, 44.2520752, -50.7471275, 44.2268829, -95.0680695, 94.9992065
42: -56.1175461, 40.1565781, -56.0577278, 40.1390800, -96.2566223, 96.2143097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1430620
time: 99.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
time: 84.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -92.4202118, 36.6817245, -92.5399475, 36.7649078, -129.1851196, 129.2216797
1: -59.2744102, 29.6923962, -59.3613129, 29.7672272, -89.0416412, 89.0537109
2: -49.3360291, 30.7054100, -49.4359779, 30.7739029, -80.1099319, 80.1413879
3: -62.2335815, 31.0667381, -62.3709946, 31.1646004, -93.3981781, 93.4377289
4: -54.7773819, 43.5166893, -54.8183937, 43.5318604, -98.3092422, 98.3350830
5: -67.1613770, 36.7612839, -67.2629852, 36.8265991, -103.9879761, 104.0242691
6: -68.9753265, 41.3595085, -68.9618912, 41.3811455, -110.3564758, 110.3213959
7: -86.4628906, 27.0098991, -86.5977631, 27.1219025, -113.5847931, 113.6076660
8: -64.5241470, 51.6350861, -64.6863098, 51.7527618, -116.2769089, 116.3213959
9: -39.0695724, 35.8085976, -39.0848312, 35.7818146, -74.8513870, 74.8934326
10: -77.0101929, 47.6595001, -77.0628510, 47.6089668, -124.6191559, 124.7223511
11: -92.8627319, 14.2013741, -93.0082703, 14.2049675, -106.4919662, 106.6424561
12: -58.7878075, 51.8803787, -58.8682404, 51.9956093, -110.7834167, 110.7486191
13: -64.0676575, 66.5264893, -63.9810677, 66.5958252, -130.6634827, 130.5075531
14: -141.9619141, 20.9177380, -142.2154999, 20.9209404, -162.8828583, 163.1332397
15: -55.7955246, 46.6429062, -55.8532028, 46.6007881, -102.3963165, 102.4961090
16: -85.3789520, 29.5299416, -85.4134064, 29.5162659, -114.8952179, 114.9433441
17: -157.3392029, 22.7043495, -157.4308167, 22.6557846, -169.5436401, 169.6831818
18: -74.8733978, 48.5643082, -75.0435944, 48.6567459, -123.5301437, 123.6079025
19: -60.3033066, 14.6933041, -60.3402824, 14.7006493, -75.0039520, 75.0335846
20: -51.6259155, 18.7022533, -51.6819992, 18.7127609, -70.3386765, 70.3842545
21: -78.3830261, 18.6106377, -78.4678192, 18.5870571, -96.9700851, 97.0784607
22: -82.4967651, 24.7530746, -82.6329346, 24.7969780, -107.2937469, 107.3860092
23: -51.8867416, 24.5374794, -51.9300537, 24.5376549, -76.4243927, 76.4675293
24: -49.9929848, 32.9975624, -50.1114464, 33.0427094, -83.0356903, 83.1090088
25: -46.2620735, 35.6423264, -46.3330154, 35.7737350, -82.0358124, 81.9753418
26: -76.7086563, 47.6147537, -76.9498215, 47.6477318, -124.3563843, 124.5645752
27: -72.9339371, 30.4089775, -73.0530701, 30.3845577, -103.3184967, 103.4620514
28: -57.9661827, 31.1847229, -58.0388641, 31.2020664, -89.1682510, 89.2235870
29: -91.4025574, 15.8589287, -91.5114136, 15.8333759, -107.2359314, 107.3703461
30: -69.6957092, 37.1639252, -69.8331299, 37.2018242, -106.8975372, 106.9970551
31: -66.0960999, 18.6338654, -66.1595001, 18.6598873, -84.7559891, 84.7933655
32: -67.3642731, 42.0468178, -67.3417740, 42.0406189, -109.4048920, 109.3885956
33: -58.6759720, 82.0697479, -58.6432114, 82.2139893, -139.8139038, 139.6352997
34: -60.4072952, 68.8641434, -60.4314499, 68.9411011, -129.0786133, 129.0211487
35: -50.6816139, 76.5169678, -50.6018219, 76.5350342, -127.2166443, 127.1187897
36: -60.5484390, 66.8659668, -60.5251846, 66.8680573, -127.4164963, 127.3911514
37: -56.2636375, 61.0419464, -56.3148689, 61.1708298, -116.7236023, 116.6416550
38: -75.3343277, 82.2243500, -75.2391815, 82.2035904, -157.5379181, 157.4635315
39: -63.9894028, 87.6000824, -63.8684311, 87.6286011, -150.9713135, 150.8312378
40: -59.2701454, 57.7009163, -59.3096390, 57.7272911, -116.9974365, 117.0105591
41: -50.8411865, 44.2520752, -50.8303909, 44.2639732, -95.1051636, 95.0824661
42: -56.1175461, 40.1565781, -56.1183090, 40.1969604, -96.3145065, 96.2748871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1430620
time: 101.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
time: 88.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 192.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 192.91
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1430620
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 192.91
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 192.91
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1430620
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 192.91
Output dim: 37, lower bound: -92.1870294, upper bound: 92.1642379
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1684341, upper bound: 92.2073375
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1768818, upper bound: 92.1393062
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.2201642, upper bound: 92.1207193
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1768818, upper bound: 92.2215954
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.2201642, upper bound: 92.2039411
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388280
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201639
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390729
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1642379, upper bound: 92.2388281
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.2073375, upper bound: 92.2201642
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 192.91
Output dim: 37, lower bound: -92.1573289, upper bound: 92.3390731
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=117.19937133789062
rel_dist={37: [-92.38023227333375, 92.38023227136748]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6597413, upper bound: 89.5817635
time: 89.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6597413, upper bound: 89.6597411
time: 113.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 204.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 204.04
Output dim: 37, lower bound: -89.6597413, upper bound: 89.5817635
IS_A2, status: Status.UNKNOWN, split count: 1, time: 204.04
Output dim: 37, lower bound: -89.6597413, upper bound: 89.6597411

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5766144, 36.7272797, -92.7582321, 36.9003296, -129.4769440, 129.4855042
1: -59.3741531, 29.7235413, -59.4987755, 29.8889580, -88.2600632, 88.2124481
2: -49.3999863, 30.7474213, -49.5271225, 30.8750706, -80.2750549, 80.2745438
3: -62.2667351, 31.1304169, -62.3929329, 31.2741623, -93.5408936, 93.5233459
4: -54.8909836, 43.5707092, -54.9806557, 43.6744614, -98.5654449, 98.5513611
5: -67.2087097, 36.8288574, -67.3317642, 36.9470024, -104.1557159, 104.1606216
6: -69.0129242, 41.5568008, -69.0951157, 41.5957832, -110.6087036, 110.6519165
7: -86.5544739, 27.0560989, -86.7188644, 27.2477875, -113.3092728, 113.2852707
8: -64.6697540, 51.6969833, -64.8558502, 51.9063072, -116.5760651, 116.5528336
9: -39.1258163, 35.8437691, -39.1785889, 35.9016991, -75.0275116, 75.0223541
10: -77.2260742, 47.7212410, -77.3168182, 47.8048325, -125.0309067, 125.0380554
11: -92.9816284, 14.2313957, -93.1463394, 14.3063030, -104.6669998, 104.7580719
12: -58.8658867, 51.9315338, -58.9799500, 52.0418930, -110.9077759, 110.9114838
13: -64.1183701, 66.7683258, -64.1669617, 66.8577728, -130.9761353, 130.9352875
14: -142.3578949, 20.9416828, -142.5885773, 21.0946770, -163.4525757, 163.5302582
15: -55.9344368, 46.6895332, -56.0348091, 46.7553253, -102.6897583, 102.7243423
16: -85.5276489, 29.5867386, -85.6254120, 29.6748753, -113.7719955, 113.7795715
17: -157.6787872, 22.7648430, -157.8168030, 22.8831139, -166.8885193, 166.9082336
18: -75.0799866, 48.5977325, -75.2313385, 48.7379074, -123.8178940, 123.8290710
19: -60.3920822, 14.7219152, -60.4491234, 14.7649746, -75.1570587, 75.1710358
20: -51.6937790, 18.7260647, -51.7735672, 18.7760601, -70.4698410, 70.4996338
21: -78.5317917, 18.6328545, -78.6122818, 18.6804543, -96.5155792, 96.5492935
22: -82.7314148, 24.7807159, -82.8361816, 24.8777390, -107.6091537, 107.6168976
23: -51.9689713, 24.5617599, -52.0270767, 24.5945568, -76.5635300, 76.5888367
24: -50.1737747, 33.0148849, -50.2734413, 33.0812759, -83.2550507, 83.2883301
25: -46.3452454, 35.6632195, -46.4771538, 35.8022003, -82.1474457, 82.1403732
26: -76.9218979, 47.6517982, -77.1235809, 47.7499771, -124.6718750, 124.7753754
27: -73.1140900, 30.4339638, -73.1814499, 30.4676971, -103.5817871, 103.6154175
28: -58.0445290, 31.2149391, -58.1206551, 31.2578163, -89.3023453, 89.3355942
29: -91.6434479, 15.8860188, -91.7362366, 15.9453735, -107.0496292, 107.0907440
30: -69.8655548, 37.1955566, -69.9638062, 37.2728653, -107.1384201, 107.1593628
31: -66.1632385, 18.6622314, -66.2646027, 18.7335796, -84.8968201, 84.9268341
32: -67.4447632, 42.2245445, -67.5085449, 42.2709427, -109.7157059, 109.7330933
33: -58.7477188, 82.2953339, -58.8959351, 82.4591980, -139.3880157, 139.3769531
34: -60.4687729, 68.9983368, -60.5910797, 69.1154785, -128.1237488, 128.1241455
35: -50.7373390, 76.7162170, -50.8097801, 76.7845612, -127.2556000, 127.2595139
36: -60.5996056, 67.1382904, -60.7323074, 67.2000427, -127.7996521, 127.8705978
37: -56.3634644, 61.1411667, -56.5628433, 61.2648849, -116.4549561, 116.5307083
38: -75.4035110, 82.5364304, -75.5077591, 82.5868378, -157.9903564, 158.0441895
39: -64.0875854, 87.8962097, -64.1732330, 87.9887009, -150.1015778, 150.0965881
40: -59.3607330, 57.8673897, -59.4722099, 57.9316673, -117.2924042, 117.3395996
41: -50.8831215, 44.4067078, -50.9647331, 44.4343987, -95.3175201, 95.3714447
42: -56.1530457, 40.2920837, -56.2144051, 40.3407364, -96.4937820, 96.5064850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5753137, upper bound: 89.5626380
time: 106.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5753137, upper bound: 89.5595160
time: 95.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -92.8168945, 36.9064636, -92.8257751, 36.9114037, -129.7283020, 129.7322388
1: -59.5403709, 29.8971672, -59.5478668, 29.9005356, -88.4014130, 88.4454727
2: -49.5681419, 30.8800659, -49.5756073, 30.8845062, -80.4526520, 80.4556732
3: -62.4353523, 31.2812347, -62.4442978, 31.2871780, -93.7225342, 93.7255325
4: -55.0073509, 43.6801758, -55.0134506, 43.6842995, -98.6916504, 98.6936264
5: -67.3706284, 36.9514656, -67.3788300, 36.9568825, -104.3275146, 104.3302917
6: -69.0903168, 41.6073341, -69.1201706, 41.6138153, -110.7041321, 110.7275085
7: -86.7659836, 27.2570705, -86.7854767, 27.2630310, -113.5104370, 113.5566483
8: -64.9211044, 51.9135361, -64.9293976, 51.9207611, -116.8418655, 116.8429337
9: -39.1908340, 35.9055328, -39.1942596, 35.9113579, -75.1021881, 75.0997925
10: -77.3383026, 47.8122673, -77.3444366, 47.8185959, -125.1568985, 125.1567078
11: -93.1560059, 14.3186169, -93.2008667, 14.3245564, -104.8467712, 104.8968964
12: -58.9852676, 52.0732651, -58.9922638, 52.0793114, -111.0645752, 111.0655289
13: -64.1752014, 66.8678436, -64.1792450, 66.8750992, -131.0502930, 131.0470886
14: -142.6594696, 21.1012630, -142.6694489, 21.1105919, -163.7700653, 163.7707062
15: -56.0562820, 46.7623138, -56.0631332, 46.7719688, -102.8282471, 102.8254471
16: -85.6339722, 29.6846275, -85.6569443, 29.6894188, -113.8776321, 113.9049530
17: -157.8517761, 22.8917389, -157.8634644, 22.9000549, -166.9898682, 167.0901184
18: -75.2397614, 48.7835312, -75.2492218, 48.7909050, -124.0306702, 124.0327530
19: -60.4546967, 14.7685261, -60.4621620, 14.7705975, -75.2252960, 75.2306900
20: -51.7836494, 18.7849102, -51.7881851, 18.7893658, -70.5730133, 70.5730972
21: -78.6214066, 18.6872406, -78.6354370, 18.6906548, -96.6120911, 96.6222763
22: -82.8503723, 24.8939972, -82.8578339, 24.9153633, -107.7657318, 107.7518311
23: -52.0333862, 24.6008911, -52.0384827, 24.6030827, -76.6364670, 76.6393738
24: -50.2813492, 33.0999718, -50.2860718, 33.1050262, -83.3863754, 83.3860474
25: -46.4842339, 35.8317833, -46.4886894, 35.8535194, -82.3377533, 82.3204727
26: -77.1396332, 47.7798233, -77.1480942, 47.7858276, -124.9254608, 124.9279175
27: -73.1921539, 30.4729614, -73.1992416, 30.4754982, -103.6676483, 103.6722031
28: -58.1272621, 31.2648582, -58.1308060, 31.2684422, -89.3957062, 89.3956604
29: -91.7566528, 15.9509907, -91.7641602, 15.9557877, -107.1653442, 107.1876221
30: -69.9721069, 37.2895775, -69.9773560, 37.2956581, -107.2677612, 107.2669373
31: -66.2725830, 18.7404671, -66.2762451, 18.7438526, -85.0164337, 85.0167084
32: -67.5052643, 42.2814713, -67.5256958, 42.2849808, -109.7902451, 109.8071671
33: -58.9039345, 82.5077591, -58.9111023, 82.5228577, -139.6161652, 139.5855408
34: -60.6011429, 69.1558380, -60.6081963, 69.1604767, -128.3094025, 128.2688904
35: -50.8169403, 76.8069000, -50.8227844, 76.8169632, -127.3487091, 127.3183746
36: -60.7409821, 67.2212677, -60.7482147, 67.2236938, -127.9646759, 127.9544373
37: -56.5746498, 61.3061066, -56.5891953, 61.3131905, -116.7189789, 116.7114944
38: -75.5191956, 82.5988159, -75.5300903, 82.6036606, -158.1228638, 158.1289062
39: -64.1811829, 88.0098114, -64.1890259, 88.0285416, -150.2237854, 150.1873016
40: -59.4822884, 57.9480896, -59.4932518, 57.9528236, -117.4351120, 117.4413452
41: -50.9663925, 44.4438477, -50.9960365, 44.4464684, -95.4128571, 95.4398804
42: -56.2136574, 40.3499680, -56.2341118, 40.3555756, -96.5692291, 96.5840759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5753137, upper bound: 89.6406047
time: 117.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5753137, upper bound: 89.6374433
time: 101.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 221.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 221.97
Output dim: 37, lower bound: -89.5753137, upper bound: 89.5626380
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 221.97
Output dim: 37, lower bound: -89.5753137, upper bound: 89.5595160
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 221.97
Output dim: 37, lower bound: -89.5753137, upper bound: 89.6406047
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 221.97
Output dim: 37, lower bound: -89.5753137, upper bound: 89.6374433

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -92.5145721, 36.7175446, -92.4953308, 36.7636414, -129.2782135, 129.2128754
1: -59.3305168, 29.7161770, -59.3283195, 29.7622337, -88.0897675, 88.0331421
2: -49.3664207, 30.7384911, -49.4000702, 30.7733269, -80.1397476, 80.1385651
3: -62.2522354, 31.1169205, -62.3313828, 31.1642056, -93.4164429, 93.4483032
4: -54.8430176, 43.5591316, -54.8018684, 43.5318375, -98.3748550, 98.3610001
5: -67.1813049, 36.8152771, -67.2275848, 36.8292885, -104.0105896, 104.0428619
6: -69.0042725, 41.5016251, -68.9705734, 41.3888893, -110.3931580, 110.4721985
7: -86.5103912, 27.0472088, -86.5582504, 27.1175346, -113.1324692, 113.1110077
8: -64.6078491, 51.6829453, -64.6335449, 51.7519608, -116.3598099, 116.3164902
9: -39.1065102, 35.8347626, -39.0778427, 35.7815704, -74.8880768, 74.9126053
10: -77.1552582, 47.7072525, -77.0615158, 47.6079369, -124.7631989, 124.7687683
11: -92.9491119, 14.2239571, -93.0104446, 14.1956749, -104.5197830, 104.6086578
12: -58.8492050, 51.9142532, -58.8710518, 51.9690552, -110.8182602, 110.7853088
13: -64.1042023, 66.7019501, -63.9778366, 66.6094894, -130.7136841, 130.6797791
14: -142.2493286, 20.9341660, -142.1835632, 20.9166756, -163.1660004, 163.1177368
15: -55.8834801, 46.6786880, -55.8444748, 46.5986481, -102.4821320, 102.5231628
16: -85.4761200, 29.5740108, -85.4186554, 29.5123787, -113.5568466, 113.5560989
17: -157.5718079, 22.7530670, -157.4280701, 22.6535759, -166.5497589, 166.4912720
18: -75.0346680, 48.5877991, -75.0564880, 48.6144028, -123.6490707, 123.6442871
19: -60.3702583, 14.7167244, -60.3437004, 14.7001743, -75.0704346, 75.0604248
20: -51.6800461, 18.7192535, -51.6791344, 18.7062454, -70.3862915, 70.3983917
21: -78.5006027, 18.6258163, -78.4743652, 18.5823784, -96.3892212, 96.4015808
22: -82.6817322, 24.7735176, -82.6431274, 24.7834930, -107.4652252, 107.4166412
23: -51.9512558, 24.5545444, -51.9324036, 24.5336819, -76.4849396, 76.4869461
24: -50.1496925, 33.0104752, -50.1236076, 33.0257263, -83.1754150, 83.1340790
25: -46.3281937, 35.6569366, -46.3347778, 35.7461815, -82.0743713, 81.9917145
26: -76.8852081, 47.6422234, -76.9566650, 47.6215858, -124.5067902, 124.5988922
27: -73.0867538, 30.4265938, -73.0620651, 30.3817272, -103.4684830, 103.4886627
28: -58.0289955, 31.2057877, -58.0406418, 31.1979485, -89.2269440, 89.2464294
29: -91.5858612, 15.8797684, -91.5154877, 15.8305645, -106.8765717, 106.8539276
30: -69.8411102, 37.1866684, -69.8436584, 37.1882057, -107.0293121, 107.0303268
31: -66.1481857, 18.6550541, -66.1585464, 18.6558762, -84.8040619, 84.8135986
32: -67.4295502, 42.1676178, -67.3536377, 42.0476990, -109.4772491, 109.5212555
33: -58.7284470, 82.2162781, -58.6423035, 82.1861496, -139.0934601, 139.0433960
34: -60.4541092, 68.9416962, -60.4277573, 68.9122849, -127.8995819, 127.9012299
35: -50.7235832, 76.6421509, -50.6003761, 76.5305481, -126.9860687, 126.9760208
36: -60.5874596, 67.0416565, -60.5217323, 66.8716278, -127.4590912, 127.5633850
37: -56.3401642, 61.1050301, -56.3133278, 61.1385956, -116.3043213, 116.2432404
38: -75.3852768, 82.4302826, -75.2347107, 82.2204285, -157.6057129, 157.6649933
39: -64.0648193, 87.7904053, -63.8705635, 87.6344223, -149.7214355, 149.6873169
40: -59.3416519, 57.8109055, -59.3090477, 57.7263222, -117.0679779, 117.1199493
41: -50.8721085, 44.3632889, -50.8329620, 44.2700958, -95.1422043, 95.1962509
42: -56.1437683, 40.2576904, -56.1226082, 40.2015991, -96.3453674, 96.3802948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5536908, upper bound: 89.4927030
time: 96.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5382481, upper bound: 89.5254798
time: 98.09 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -92.5761642, 36.7271729, -92.7511292, 36.8986969, -129.4748535, 129.4783020
1: -59.3738022, 29.7234631, -59.4933167, 29.8876209, -88.2584991, 88.1795654
2: -49.3997726, 30.7473335, -49.5246544, 30.8735828, -80.2733536, 80.2719879
3: -62.2663116, 31.1302662, -62.3863564, 31.2719173, -93.5382309, 93.5166245
4: -54.8905830, 43.5705643, -54.9742699, 43.6722183, -98.5628052, 98.5448303
5: -67.2084351, 36.8287086, -67.3270950, 36.9449043, -104.1533356, 104.1558075
6: -69.0128174, 41.5563049, -69.0933456, 41.5881310, -110.6009521, 110.6496506
7: -86.5540848, 27.0559692, -86.7144623, 27.2460461, -113.3072205, 113.2533722
8: -64.6692505, 51.6968460, -64.8478775, 51.9039078, -116.5731583, 116.5447235
9: -39.1255913, 35.8436699, -39.1755524, 35.8999672, -75.0255585, 75.0192261
10: -77.2255249, 47.7210388, -77.3081055, 47.8020134, -125.0275421, 125.0291443
11: -92.9813156, 14.2313004, -93.1414185, 14.3048687, -104.6649628, 104.7166138
12: -58.8657265, 51.9311104, -58.9775085, 52.0353432, -110.9010696, 110.9086151
13: -64.1181946, 66.7677765, -64.1642380, 66.8487473, -130.9669495, 130.9320068
14: -142.3569336, 20.9415874, -142.5735626, 21.0933933, -163.4503326, 163.5151520
15: -55.9340401, 46.6894150, -56.0283928, 46.7534332, -102.6874695, 102.7178040
16: -85.5271912, 29.5866051, -85.6181641, 29.6727238, -113.7693176, 113.7374573
17: -157.6779022, 22.7647018, -157.8032379, 22.8807411, -166.8851471, 166.8170471
18: -75.0795670, 48.5976219, -75.2250519, 48.7361145, -123.8156815, 123.8226776
19: -60.3918190, 14.7218456, -60.4453697, 14.7639389, -75.1557617, 75.1672134
20: -51.6936188, 18.7259827, -51.7711334, 18.7747688, -70.4683838, 70.4971161
21: -78.5314331, 18.6327839, -78.6068726, 18.6793766, -96.5140839, 96.5268707
22: -82.7309189, 24.7806377, -82.8286514, 24.8765869, -107.6075058, 107.6092911
23: -51.9687729, 24.5616875, -52.0238647, 24.5932846, -76.5620575, 76.5855560
24: -50.1733856, 33.0148201, -50.2673912, 33.0803909, -83.2537766, 83.2822113
25: -46.3450508, 35.6631393, -46.4740181, 35.8010178, -82.1460724, 82.1371613
26: -76.9215088, 47.6516800, -77.1182251, 47.7481384, -124.6696472, 124.7699051
27: -73.1136780, 30.4338722, -73.1752243, 30.4663658, -103.5800476, 103.6091003
28: -58.0443306, 31.2148380, -58.1174736, 31.2562485, -89.3005829, 89.3323135
29: -91.6428680, 15.8859596, -91.7271881, 15.9442692, -107.0477905, 107.0366821
30: -69.8652267, 37.1954651, -69.9586639, 37.2713623, -107.1365891, 107.1541290
31: -66.1630707, 18.6621609, -66.2619171, 18.7323551, -84.8954239, 84.9240799
32: -67.4445496, 42.2240448, -67.5054169, 42.2629395, -109.7074890, 109.7294617
33: -58.7474861, 82.2946930, -58.8927307, 82.4490891, -139.3599243, 139.3730621
34: -60.4685936, 68.9978790, -60.5884018, 69.1080170, -128.0873260, 128.1208344
35: -50.7371521, 76.7156677, -50.8069839, 76.7755737, -127.2232437, 127.2561188
36: -60.5994415, 67.1375427, -60.7298965, 67.1879425, -127.7746048, 127.8674393
37: -56.3631897, 61.1408539, -56.5585709, 61.2599030, -116.4386597, 116.5260391
38: -75.4033508, 82.5355377, -75.5048828, 82.5727081, -157.9760590, 158.0404205
39: -64.0873032, 87.8953247, -64.1688232, 87.9749832, -150.0554810, 150.0913086
40: -59.3605118, 57.8669128, -59.4686546, 57.9243164, -117.2848282, 117.3355713
41: -50.8829918, 44.4063225, -50.9627647, 44.4281540, -95.3111420, 95.3690872
42: -56.1529503, 40.2917404, -56.2129898, 40.3352356, -96.4881897, 96.5047302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6374435, upper bound: 89.4973256
time: 78.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6374435, upper bound: 89.5595160
time: 97.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -92.7548676, 36.8967133, -92.5629044, 36.7746735, -129.5295410, 129.4596252
1: -59.4967422, 29.8898144, -59.3774071, 29.7738380, -88.2311249, 88.2661972
2: -49.5345840, 30.8711319, -49.4485741, 30.7827473, -80.3173294, 80.3197021
3: -62.4208412, 31.2677269, -62.3827591, 31.1772118, -93.5980530, 93.6504822
4: -54.9593811, 43.6686020, -54.8346748, 43.5416870, -98.5010681, 98.5032806
5: -67.3432312, 36.9378853, -67.2746658, 36.8391647, -104.1823959, 104.2125549
6: -69.0816650, 41.5522003, -68.9956207, 41.4069023, -110.4885712, 110.5478210
7: -86.7218933, 27.2481689, -86.6248474, 27.1327782, -113.3336334, 113.3824081
8: -64.8592072, 51.8994751, -64.7070999, 51.7664185, -116.6256256, 116.6065750
9: -39.1715202, 35.8965034, -39.0935059, 35.7912292, -74.9627533, 74.9900055
10: -77.2675018, 47.7982864, -77.0891724, 47.6216927, -124.8891907, 124.8874588
11: -93.1234589, 14.3112240, -93.0649643, 14.2139225, -104.6995850, 104.7474747
12: -58.9685745, 52.0559845, -58.8833466, 52.0064507, -110.9750214, 110.9393311
13: -64.1609955, 66.8014526, -63.9901505, 66.6268005, -130.7877960, 130.7915955
14: -142.5509491, 21.0937424, -142.2644348, 20.9325581, -163.4835052, 163.3581848
15: -56.0053444, 46.7514496, -55.8727646, 46.6152954, -102.6206360, 102.6242142
16: -85.5824661, 29.6718979, -85.4502106, 29.5269279, -113.6624374, 113.6814804
17: -157.7448425, 22.8799667, -157.4746857, 22.6704941, -166.6510925, 166.6732025
18: -75.1944580, 48.7735634, -75.0744019, 48.6673927, -123.8618469, 123.8479614
19: -60.4328880, 14.7633314, -60.3567200, 14.7058086, -75.1386948, 75.1200485
20: -51.7699165, 18.7780991, -51.6937561, 18.7195663, -70.4894867, 70.4718552
21: -78.5902252, 18.6801987, -78.4975586, 18.5925865, -96.4857635, 96.4744797
22: -82.8006744, 24.8867989, -82.6647720, 24.8211136, -107.6217880, 107.5515747
23: -52.0156860, 24.5936832, -51.9437866, 24.5422173, -76.5579071, 76.5374680
24: -50.2572937, 33.0955658, -50.1361961, 33.0494995, -83.3067932, 83.2317657
25: -46.4671860, 35.8254967, -46.3462677, 35.7975082, -82.2646942, 82.1717682
26: -77.1029892, 47.7702713, -76.9811554, 47.6574554, -124.7604446, 124.7514267
27: -73.1648407, 30.4655590, -73.0798340, 30.3895473, -103.5543900, 103.5453949
28: -58.1117287, 31.2556973, -58.0507698, 31.2085876, -89.3203125, 89.3064651
29: -91.6990662, 15.9447021, -91.5434189, 15.8409767, -106.9923019, 106.9507751
30: -69.9476624, 37.2806625, -69.8571777, 37.2110138, -107.1586761, 107.1378403
31: -66.2575455, 18.7332840, -66.1701660, 18.6661568, -84.9237061, 84.9034500
32: -67.4900360, 42.2245483, -67.3707809, 42.0616913, -109.5517273, 109.5953293
33: -58.8847008, 82.4287186, -58.6575241, 82.2498016, -139.3216248, 139.2519989
34: -60.5864906, 69.0991745, -60.4448433, 68.9572906, -128.0852356, 128.0459900
35: -50.8031960, 76.7328720, -50.6133347, 76.5629959, -127.0791321, 127.0348816
36: -60.7288322, 67.1246185, -60.5376358, 66.8952637, -127.6240997, 127.6470871
37: -56.5513725, 61.2699852, -56.3396645, 61.1869164, -116.5683289, 116.4240265
38: -75.5009689, 82.4926910, -75.2570038, 82.2372665, -157.7382355, 157.7496948
39: -64.1584015, 87.9040375, -63.8863640, 87.6742859, -149.8436127, 149.7780609
40: -59.4632339, 57.8916016, -59.3301048, 57.7474937, -117.2107239, 117.2217102
41: -50.9553909, 44.4004326, -50.8642731, 44.2821503, -95.2375412, 95.2647095
42: -56.2043457, 40.3155899, -56.1422997, 40.2164154, -96.4207611, 96.4578857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5536908, upper bound: 89.5706586
time: 103.46 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5382481, upper bound: 89.6033126
time: 139.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -92.8164520, 36.9063530, -92.8186646, 36.9097595, -129.7262115, 129.7250214
1: -59.5400162, 29.8970757, -59.5424004, 29.8992157, -88.3998413, 88.4125977
2: -49.5679359, 30.8799782, -49.5731392, 30.8830185, -80.4509583, 80.4531174
3: -62.4349327, 31.2810936, -62.4377136, 31.2849388, -93.7198715, 93.7188110
4: -55.0069504, 43.6800308, -55.0070686, 43.6820602, -98.6890106, 98.6871033
5: -67.3703461, 36.9513321, -67.3741760, 36.9547920, -104.3251343, 104.3255081
6: -69.0902023, 41.6068573, -69.1184082, 41.6061783, -110.6963806, 110.7252655
7: -86.7656021, 27.2569599, -86.7810669, 27.2612667, -113.5083618, 113.5247421
8: -64.9206085, 51.9133797, -64.9214096, 51.9183884, -116.8389969, 116.8347931
9: -39.1906204, 35.9054184, -39.1912346, 35.9096451, -75.1002655, 75.0966492
10: -77.3377533, 47.8120880, -77.3357391, 47.8157463, -125.1535034, 125.1478271
11: -93.1556854, 14.3185310, -93.1959381, 14.3231297, -104.8447189, 104.8554611
12: -58.9851112, 52.0728683, -58.9898224, 52.0727310, -111.0578461, 111.0626907
13: -64.1750183, 66.8672791, -64.1765442, 66.8660889, -131.0411072, 131.0438232
14: -142.6585083, 21.1011868, -142.6544037, 21.1093216, -163.7678223, 163.7555847
15: -56.0558891, 46.7621918, -56.0566978, 46.7700806, -102.8259735, 102.8188934
16: -85.6335144, 29.6844826, -85.6497116, 29.6872711, -113.8749237, 113.8628387
17: -157.8509216, 22.8916168, -157.8498993, 22.8976936, -166.9865112, 166.9989166
18: -75.2393494, 48.7834167, -75.2429352, 48.7891121, -124.0284576, 124.0263519
19: -60.4544296, 14.7684507, -60.4584084, 14.7695675, -75.2239990, 75.2268600
20: -51.7835007, 18.7848263, -51.7857590, 18.7880516, -70.5715485, 70.5705872
21: -78.6210480, 18.6871624, -78.6300354, 18.6895752, -96.6105957, 96.5998001
22: -82.8498688, 24.8939133, -82.8503189, 24.9141922, -107.7640610, 107.7442322
23: -52.0331879, 24.6008129, -52.0352631, 24.6017952, -76.6349792, 76.6360779
24: -50.2809677, 33.0999146, -50.2800102, 33.1041641, -83.3851318, 83.3799286
25: -46.4840355, 35.8317108, -46.4855118, 35.8523483, -82.3363800, 82.3172226
26: -77.1392517, 47.7797127, -77.1427612, 47.7839737, -124.9232254, 124.9224701
27: -73.1917572, 30.4728737, -73.1930084, 30.4741688, -103.6659241, 103.6658783
28: -58.1270676, 31.2647667, -58.1275826, 31.2668571, -89.3939209, 89.3923492
29: -91.7560730, 15.9509010, -91.7551117, 15.9546757, -107.1634827, 107.1335678
30: -69.9717789, 37.2894821, -69.9722137, 37.2941551, -107.2659302, 107.2616959
31: -66.2724152, 18.7403812, -66.2735443, 18.7426376, -85.0150528, 85.0139236
32: -67.5050659, 42.2809525, -67.5225677, 42.2769547, -109.7820206, 109.8035202
33: -58.9037361, 82.5071259, -58.9079132, 82.5127487, -139.5881195, 139.5816956
34: -60.6009521, 69.1553650, -60.6054535, 69.1530151, -128.2730103, 128.2655945
35: -50.8167572, 76.8063354, -50.8199425, 76.8079834, -127.3162766, 127.3149872
36: -60.7408524, 67.2205124, -60.7458038, 67.2116013, -127.9459000, 127.9512634
37: -56.5743713, 61.3057938, -56.5848961, 61.3082008, -116.7026291, 116.7068634
38: -75.5190277, 82.5979080, -75.5272141, 82.5895844, -158.1086121, 158.1251221
39: -64.1808777, 88.0089340, -64.1846619, 88.0148010, -150.1776733, 150.1820374
40: -59.4820518, 57.9476395, -59.4897308, 57.9454575, -117.4275055, 117.4373703
41: -50.9662666, 44.4434547, -50.9940796, 44.4402351, -95.4065018, 95.4375305
42: -56.2135620, 40.3496208, -56.2326927, 40.3500633, -96.5636292, 96.5823135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1598

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6374435, upper bound: 89.5753135
time: 141.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6374435, upper bound: 89.6374433
time: 130.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 273.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.5536908, upper bound: 89.4927030
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.5382481, upper bound: 89.5254798
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.6374435, upper bound: 89.4973256
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.6374435, upper bound: 89.5595160
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.5536908, upper bound: 89.5706586
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.5382481, upper bound: 89.6033126
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.6374435, upper bound: 89.5753135
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 273.99
Output dim: 37, lower bound: -89.6374435, upper bound: 89.6374433

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -92.4013290, 36.6787148, -92.4666443, 36.7537689, -129.1550903, 129.1453552
1: -59.2609863, 29.6900768, -59.3107414, 29.7555962, -88.0164719, 87.9880829
2: -49.3257294, 30.7026215, -49.3896828, 30.7643032, -80.0900345, 80.0923004
3: -62.2291260, 31.0625229, -62.3256149, 31.1505127, -93.3796387, 93.3881378
4: -54.7626495, 43.5131073, -54.7812614, 43.5201797, -98.2828293, 98.2943726
5: -67.1529312, 36.7570648, -67.2204895, 36.8146744, -103.9676056, 103.9775543
6: -68.9726410, 41.3425560, -68.9626007, 41.3493538, -110.3219910, 110.3051605
7: -86.4493561, 27.0071297, -86.5427017, 27.1074276, -113.0610809, 113.0526657
8: -64.5051727, 51.6307602, -64.6078949, 51.7388229, -116.2439957, 116.2386551
9: -39.0636597, 35.8058167, -39.0670929, 35.7742233, -74.8378830, 74.8729095
10: -76.9883728, 47.6552277, -77.0201263, 47.5948105, -124.5831833, 124.6753540
11: -92.8527527, 14.1990910, -92.9862976, 14.1894436, -104.4036484, 104.5559845
12: -58.7826385, 51.8750763, -58.8543739, 51.9592590, -110.7418976, 110.7294464
13: -64.0633240, 66.5063019, -63.9675751, 66.5608521, -130.6241760, 130.4738770
14: -141.9286499, 20.9154778, -142.1037445, 20.9119778, -162.8406219, 163.0192261
15: -55.7798538, 46.6395569, -55.8183975, 46.5887108, -102.3685608, 102.4579544
16: -85.3631134, 29.5260277, -85.3903580, 29.5003433, -113.4284134, 113.4769211
17: -157.3063354, 22.7007141, -157.3620300, 22.6403923, -166.2560577, 166.3681641
18: -74.8594971, 48.5612450, -75.0128021, 48.6077080, -123.4672089, 123.5740509
19: -60.2966309, 14.6917191, -60.3253059, 14.6938553, -74.9904861, 75.0170288
20: -51.6216888, 18.7001400, -51.6643219, 18.7014351, -70.3231201, 70.3644638
21: -78.3734894, 18.6084747, -78.4422531, 18.5780144, -96.2520905, 96.3495178
22: -82.4815521, 24.7508736, -82.5932312, 24.7777767, -107.2593307, 107.3441010
23: -51.8812714, 24.5352707, -51.9147682, 24.5288315, -76.4101028, 76.4500427
24: -49.9856758, 32.9962082, -50.0825119, 33.0221481, -83.0078278, 83.0787201
25: -46.2568169, 35.6403885, -46.3166962, 35.7419968, -81.9988098, 81.9570847
26: -76.6974182, 47.6118088, -76.9098053, 47.6139565, -124.3113708, 124.5216141
27: -72.9257050, 30.4066849, -73.0217285, 30.3767262, -103.3024292, 103.4284134
28: -57.9614830, 31.1818905, -58.0235100, 31.1919479, -89.1534271, 89.2053986
29: -91.3849487, 15.8570080, -91.4653625, 15.8247910, -106.6548462, 106.7755737
30: -69.6883316, 37.1611824, -69.8051605, 37.1818314, -106.8701630, 106.9663391
31: -66.0914383, 18.6316509, -66.1442261, 18.6499634, -84.7414017, 84.7758789
32: -67.3595886, 42.0293770, -67.3360901, 42.0118103, -109.3713989, 109.3654633
33: -58.6700401, 82.0454559, -58.6277008, 82.1437073, -138.9900818, 138.8525391
34: -60.4027557, 68.8467255, -60.4147568, 68.8886185, -127.8192978, 127.7872314
35: -50.6773796, 76.4942169, -50.5887337, 76.4938507, -126.9003448, 126.8103409
36: -60.5446892, 66.8363495, -60.5109596, 66.8208160, -127.3655090, 127.3450623
37: -56.2564468, 61.0308304, -56.2922821, 61.1201591, -116.2004852, 116.1458054
38: -75.3287201, 82.1918259, -75.2204590, 82.1613235, -157.4900513, 157.4122925
39: -63.9824142, 87.5676727, -63.8499069, 87.5791626, -149.5807800, 149.4365234
40: -59.2642784, 57.6835480, -59.2896118, 57.6946449, -116.9589233, 116.9731598
41: -50.8377991, 44.2387848, -50.8243370, 44.2382278, -95.0760269, 95.0631256
42: -56.1146774, 40.1460228, -56.1153526, 40.1732330, -96.2879105, 96.2613754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5157106, upper bound: 89.4927030
time: 86.05 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5157106, upper bound: 89.4927030
time: 101.30 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -92.6575089, 36.8042336, -92.4869614, 36.7614441, -129.4189453, 129.2911987
1: -59.3847809, 29.8756886, -59.3249664, 29.7606697, -88.1485748, 88.1878815
2: -49.3946915, 30.8005905, -49.3943863, 30.7715988, -80.1662903, 80.1949768
3: -62.3012733, 31.2041397, -62.3294640, 31.1619263, -93.4631958, 93.5335999
4: -54.8675385, 43.7125854, -54.7951126, 43.5294571, -98.3969955, 98.5076981
5: -67.2195358, 36.8869019, -67.2252350, 36.8270264, -104.0465622, 104.1121368
6: -69.2068481, 41.5458031, -68.9681931, 41.3835678, -110.5904160, 110.5139923
7: -86.5217590, 27.1738567, -86.5513458, 27.1158714, -113.1505432, 113.2195053
8: -64.6235046, 51.8392410, -64.6263733, 51.7495193, -116.3730240, 116.4656143
9: -39.2124672, 35.9692535, -39.0758896, 35.7798386, -74.9923096, 75.0451431
10: -77.1849518, 47.9505310, -77.0562973, 47.6051292, -124.7900848, 125.0068283
11: -93.0054016, 14.4202023, -93.0064316, 14.1942806, -104.5748978, 104.8048477
12: -58.9120827, 51.9655266, -58.8685760, 51.9654465, -110.8775330, 110.8341064
13: -64.4477997, 66.7448502, -63.9755020, 66.6018600, -131.0496521, 130.7203522
14: -142.3302002, 21.2296925, -142.1726990, 20.9153366, -163.2455444, 163.4023895
15: -55.9157791, 46.8780632, -55.8412285, 46.5964966, -102.5122757, 102.7192917
16: -85.5704727, 29.7682114, -85.4141464, 29.5104313, -113.6473083, 113.7460709
17: -157.6139832, 23.1053810, -157.4199371, 22.6510429, -166.5809937, 166.8363342
18: -75.0742493, 48.8341522, -75.0502319, 48.6126595, -123.6869049, 123.8843842
19: -60.4271278, 14.7960815, -60.3406410, 14.6988068, -75.1259308, 75.1367188
20: -51.7314835, 18.8558064, -51.6758499, 18.7050819, -70.4365692, 70.5316544
21: -78.5576324, 18.7869797, -78.4696198, 18.5810928, -96.4507370, 96.5561829
22: -82.7228699, 24.9320984, -82.6371918, 24.7819748, -107.5048447, 107.5692902
23: -52.0172577, 24.6603031, -51.9288864, 24.5323200, -76.5495758, 76.5891876
24: -50.3058624, 33.1498260, -50.1170807, 33.0248947, -83.3307571, 83.2669067
25: -46.4722824, 35.7582817, -46.3311119, 35.7450218, -82.2173004, 82.0893936
26: -76.9573364, 47.9228783, -76.9501114, 47.6194725, -124.5768127, 124.8729858
27: -73.1408081, 30.6165886, -73.0555420, 30.3803406, -103.5211487, 103.6721344
28: -58.0763359, 31.2988625, -58.0368042, 31.1964951, -89.2728271, 89.3356628
29: -91.6418610, 16.0636578, -91.5093994, 15.8291073, -106.9329681, 107.0336685
30: -69.9196701, 37.3709373, -69.8361206, 37.1866798, -107.1063538, 107.2070618
31: -66.2420654, 18.7569237, -66.1554565, 18.6544952, -84.8965607, 84.9123840
32: -67.6293182, 42.2479706, -67.3505554, 42.0433273, -109.6726456, 109.5985260
33: -59.1013985, 82.2237320, -58.6391258, 82.1807098, -139.4601746, 139.0435333
34: -60.6497383, 68.9734344, -60.4244308, 68.9090652, -128.0897827, 127.9258881
35: -51.0369263, 76.6492538, -50.5975800, 76.5257339, -127.2933350, 126.9746628
36: -60.8788223, 67.0415955, -60.5197258, 66.8651352, -127.7439575, 127.5584412
37: -56.6781082, 61.1104317, -56.3092461, 61.1358109, -116.6421356, 116.2424622
38: -75.7676697, 82.4557190, -75.2320557, 82.2121964, -157.9798584, 157.6877747
39: -64.5160370, 87.7844772, -63.8667450, 87.6269150, -150.1672058, 149.6702881
40: -59.5406837, 57.8421516, -59.3050308, 57.7225838, -117.2632675, 117.1471863
41: -51.0876656, 44.3980789, -50.8308716, 44.2635727, -95.3512421, 95.2289505
42: -56.2746277, 40.3054314, -56.1210365, 40.1970825, -96.4717102, 96.4264679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5003211, upper bound: 89.5254798
time: 108.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5003211, upper bound: 89.5254798
time: 87.09 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -92.3136139, 36.5905609, -92.7511292, 36.8986969, -129.2123108, 129.3416901
1: -59.2036133, 29.5968208, -59.4933167, 29.8876209, -88.0867310, 88.0815964
2: -49.2729111, 30.6456985, -49.5246544, 30.8735828, -80.1464920, 80.1703491
3: -62.2051926, 31.0205917, -62.3863564, 31.2719173, -93.4771118, 93.4069519
4: -54.7121887, 43.4280968, -54.9742699, 43.6722183, -98.3844070, 98.4023666
5: -67.1045074, 36.7111931, -67.3270950, 36.9449043, -104.0494080, 104.0382843
6: -68.8884354, 41.3498650, -69.0933456, 41.5881310, -110.4765625, 110.4432068
7: -86.3938446, 26.9258995, -86.7144623, 27.2460461, -113.1423187, 113.1513519
8: -64.4474335, 51.5426826, -64.8478775, 51.9039078, -116.3513412, 116.3905640
9: -39.0250511, 35.7236633, -39.1755524, 35.8999672, -74.9250183, 74.8992157
10: -76.9707565, 47.5243340, -77.3081055, 47.8020134, -124.7727661, 124.8324432
11: -92.8457031, 14.1207809, -93.1414185, 14.3048687, -104.5238800, 104.6376419
12: -58.7570419, 51.8586197, -58.9775085, 52.0353432, -110.7923889, 110.8361282
13: -63.9292984, 66.5201035, -64.1642380, 66.8487473, -130.7780457, 130.6843414
14: -141.9527435, 20.7636433, -142.5735626, 21.0933933, -163.0461426, 163.3372040
15: -55.7440834, 46.5329399, -56.0283928, 46.7534332, -102.4975128, 102.5613327
16: -85.3208466, 29.4242210, -85.6181641, 29.6727238, -113.5591583, 113.6101379
17: -157.2899323, 22.5353851, -157.8032379, 22.8807411, -166.4806671, 166.6651459
18: -74.9051056, 48.4742432, -75.2250519, 48.7361145, -123.6412201, 123.6992950
19: -60.2866096, 14.6571159, -60.4453697, 14.7639389, -75.0505524, 75.1024857
20: -51.5993767, 18.6562786, -51.7711334, 18.7747688, -70.3741455, 70.4274139
21: -78.3938751, 18.5347767, -78.6068726, 18.6793766, -96.3736877, 96.4464493
22: -82.5383453, 24.6864796, -82.8286514, 24.8765869, -107.4149323, 107.5151291
23: -51.8742943, 24.5009155, -52.0238647, 24.5932846, -76.4675751, 76.5247803
24: -50.0239906, 32.9593582, -50.2673912, 33.0803909, -83.1043854, 83.2267456
25: -46.2029533, 35.6072159, -46.4740181, 35.8010178, -82.0039673, 82.0812378
26: -76.7549591, 47.5233841, -77.1182251, 47.7481384, -124.5030975, 124.6416092
27: -72.9947052, 30.3480320, -73.1752243, 30.4663658, -103.4610748, 103.5232544
28: -57.9645309, 31.1550789, -58.1174736, 31.2562485, -89.2207794, 89.2725525
29: -91.4226532, 15.7712669, -91.7271881, 15.9442692, -106.8180237, 106.9666595
30: -69.7454376, 37.1109428, -69.9586639, 37.2713623, -107.0167999, 107.0696106
31: -66.0572739, 18.5845509, -66.2619171, 18.7323551, -84.7896271, 84.8464661
32: -67.2898712, 42.0011368, -67.5054169, 42.2629395, -109.5528107, 109.5065536
33: -58.4940872, 82.0222321, -58.8927307, 82.4490891, -139.1238708, 139.0984955
34: -60.3054276, 68.7951355, -60.5884018, 69.1080170, -127.9510498, 127.9124908
35: -50.5279999, 76.4622040, -50.8069839, 76.7755737, -127.0377045, 127.0007858
36: -60.3890152, 66.8098526, -60.7298965, 67.1879425, -127.5769577, 127.5388794
37: -56.1139832, 61.0148468, -56.5585709, 61.2599030, -116.1989594, 116.3990402
38: -75.1305008, 82.1700287, -75.5048828, 82.5727081, -157.7032166, 157.6749115
39: -63.7851028, 87.5419006, -64.1688232, 87.9749832, -149.7852783, 149.7345734
40: -59.1976013, 57.6620064, -59.4686546, 57.9243164, -117.1219177, 117.1306610
41: -50.7513390, 44.2424469, -50.9627647, 44.4281540, -95.1794891, 95.2052155
42: -56.0612755, 40.1529388, -56.2129898, 40.3352356, -96.3965149, 96.3659286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5052296, upper bound: 89.4757030
time: 93.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5382481, upper bound: 89.4603084
time: 91.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -92.5695038, 36.7256088, -92.7511292, 36.8986969, -129.4682007, 129.4767456
1: -59.3686676, 29.7222290, -59.4933167, 29.8876209, -88.2259445, 88.1782990
2: -49.3975220, 30.7459259, -49.5246544, 30.8735828, -80.2711029, 80.2705841
3: -62.2601357, 31.1281586, -62.3863564, 31.2719173, -93.5320511, 93.5145111
4: -54.8846397, 43.5684624, -54.9742699, 43.6722183, -98.5568542, 98.5427322
5: -67.2040558, 36.8267288, -67.3270950, 36.9449043, -104.1489563, 104.1538239
6: -69.0111694, 41.5491295, -69.0933456, 41.5881310, -110.5993042, 110.6424713
7: -86.5500946, 27.0543671, -86.7144623, 27.2460461, -113.2757111, 113.2517090
8: -64.6617661, 51.6946106, -64.8478775, 51.9039078, -116.5656738, 116.5424881
9: -39.1227798, 35.8420563, -39.1755524, 35.8999672, -75.0227509, 75.0176086
10: -77.2173767, 47.7183800, -77.3081055, 47.8020134, -125.0193939, 125.0264893
11: -92.9767303, 14.2299519, -93.1414185, 14.3048687, -104.6240845, 104.7151794
12: -58.8634491, 51.9249840, -58.9775085, 52.0353432, -110.8987885, 110.9024963
13: -64.1156464, 66.7593079, -64.1642380, 66.8487473, -130.9643860, 130.9235535
14: -142.3428497, 20.9403877, -142.5735626, 21.0933933, -163.4362488, 163.5139465
15: -55.9280281, 46.6876526, -56.0283928, 46.7534332, -102.6814575, 102.7160492
16: -85.5204010, 29.5845795, -85.6181641, 29.6727238, -113.7278061, 113.7353516
17: -157.6652222, 22.7625427, -157.8032379, 22.8807411, -166.7950439, 166.8148041
18: -75.0736847, 48.5959663, -75.2250519, 48.7361145, -123.8097992, 123.8210144
19: -60.3883247, 14.7208900, -60.4453697, 14.7639389, -75.1522675, 75.1662598
20: -51.6913490, 18.7247658, -51.7711334, 18.7747688, -70.4661179, 70.4958954
21: -78.5263824, 18.6318035, -78.6068726, 18.6793766, -96.4920883, 96.5258636
22: -82.7238922, 24.7795563, -82.8286514, 24.8765869, -107.6004791, 107.6082077
23: -51.9657516, 24.5604820, -52.0238647, 24.5932846, -76.5590363, 76.5843506
24: -50.1676903, 33.0140190, -50.2673912, 33.0803909, -83.2480774, 83.2814102
25: -46.3420906, 35.6620560, -46.4740181, 35.8010178, -82.1431122, 82.1360779
26: -76.9165802, 47.6499405, -77.1182251, 47.7481384, -124.6647186, 124.7681656
27: -73.1078491, 30.4326401, -73.1752243, 30.4663658, -103.5742188, 103.6078644
28: -58.0413132, 31.2133579, -58.1174736, 31.2562485, -89.2975616, 89.3308334
29: -91.6343842, 15.8849487, -91.7271881, 15.9442692, -106.9944611, 107.0355682
30: -69.8604126, 37.1940765, -69.9586639, 37.2713623, -107.1317749, 107.1527405
31: -66.1605682, 18.6610165, -66.2619171, 18.7323551, -84.8929214, 84.9229355
32: -67.4416046, 42.2165298, -67.5054169, 42.2629395, -109.7045441, 109.7219467
33: -58.7444916, 82.2852097, -58.8927307, 82.4490891, -139.3569336, 139.3458405
34: -60.4660187, 68.9908752, -60.5884018, 69.1080170, -128.0847015, 128.0850830
35: -50.7345238, 76.7072296, -50.8069839, 76.7755737, -127.2206039, 127.2245102
36: -60.5971680, 67.1262054, -60.7298965, 67.1879425, -127.7723770, 127.8405609
37: -56.3591576, 61.1361694, -56.5585709, 61.2599030, -116.4345703, 116.5103149
38: -75.4006653, 82.5223236, -75.5048828, 82.5727081, -157.9733734, 158.0272064
39: -64.0832062, 87.8824692, -64.1688232, 87.9749832, -150.0513916, 150.0463867
40: -59.3571739, 57.8600349, -59.4686546, 57.9243164, -117.2814941, 117.3286896
41: -50.8811607, 44.4004860, -50.9627647, 44.4281540, -95.3093109, 95.3632507
42: -56.1516380, 40.2865906, -56.2129898, 40.3352356, -96.4868774, 96.4995804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5052296, upper bound: 89.5378726
time: 101.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5382481, upper bound: 89.5236112
time: 107.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -92.6415863, 36.8578796, -92.5341873, 36.7648315, -129.4064178, 129.3920593
1: -59.4272270, 29.8637009, -59.3598213, 29.7671928, -88.1578674, 88.2211456
2: -49.4938889, 30.8352356, -49.4381752, 30.7737293, -80.2676163, 80.2734070
3: -62.3977509, 31.2133026, -62.3769875, 31.1635361, -93.5612869, 93.5902863
4: -54.8790283, 43.6225853, -54.8140678, 43.5300064, -98.4090347, 98.4366531
5: -67.3148499, 36.8796501, -67.2675552, 36.8245392, -104.1393890, 104.1472015
6: -69.0500031, 41.3931732, -68.9876556, 41.3673706, -110.4173737, 110.3808289
7: -86.6608734, 27.2081299, -86.6092911, 27.1226254, -113.2622528, 113.3240509
8: -64.7565384, 51.8472672, -64.6814423, 51.7532768, -116.5098114, 116.5287094
9: -39.1286545, 35.8675613, -39.0827637, 35.7838821, -74.9125366, 74.9503250
10: -77.1006546, 47.7462349, -77.0477753, 47.6085701, -124.7092285, 124.7940063
11: -93.0271606, 14.2863445, -93.0408096, 14.2076740, -104.5835114, 104.6948090
12: -58.9020042, 52.0168114, -58.8666840, 51.9966278, -110.8986359, 110.8834991
13: -64.1200333, 66.6057968, -63.9798851, 66.5781708, -130.6982117, 130.5856781
14: -142.2303314, 21.0750313, -142.1846008, 20.9278736, -163.1582031, 163.2596283
15: -55.9017944, 46.7122574, -55.8466759, 46.6053619, -102.5071564, 102.5589294
16: -85.4694977, 29.6239471, -85.4219284, 29.5148926, -113.5340271, 113.6023560
17: -157.4794006, 22.8276062, -157.4086609, 22.6573486, -166.3574371, 166.5500793
18: -75.0193481, 48.7470207, -75.0307159, 48.6607132, -123.6800613, 123.7777405
19: -60.3592567, 14.7383471, -60.3383179, 14.6994858, -75.0587463, 75.0766678
20: -51.7115517, 18.7589264, -51.6789474, 18.7147408, -70.4262924, 70.4378738
21: -78.4631805, 18.6628780, -78.4654312, 18.5882492, -96.3486633, 96.4224548
22: -82.6005936, 24.8641510, -82.6148682, 24.8153992, -107.4159927, 107.4790192
23: -51.9457016, 24.5743980, -51.9261398, 24.5373669, -76.4830704, 76.5005341
24: -50.0933723, 33.0812988, -50.0951157, 33.0459290, -83.1392975, 83.1764145
25: -46.3958397, 35.8089218, -46.3282051, 35.7933311, -82.1891708, 82.1371307
26: -76.9152603, 47.7398529, -76.9342880, 47.6498184, -124.5650787, 124.6741409
27: -73.0037994, 30.4456272, -73.0395050, 30.3845367, -103.3883362, 103.4851303
28: -58.0442543, 31.2317924, -58.0336456, 31.2025738, -89.2468262, 89.2654419
29: -91.4981308, 15.9219093, -91.4932556, 15.8352356, -106.7707214, 106.8723907
30: -69.7949524, 37.2551498, -69.8186798, 37.2046127, -106.9995651, 107.0738297
31: -66.2007828, 18.7099075, -66.1558380, 18.6602535, -84.8610382, 84.8657455
32: -67.4200745, 42.0863800, -67.3532257, 42.0258026, -109.4458771, 109.4396057
33: -58.8263359, 82.2579498, -58.6429176, 82.2073746, -139.2183533, 139.0612335
34: -60.5350723, 69.0041962, -60.4318352, 68.9336243, -128.0049133, 127.9319763
35: -50.7569275, 76.5849686, -50.6016769, 76.5262604, -126.9933472, 126.8692322
36: -60.6861229, 66.9192963, -60.5268707, 66.8444595, -127.5305786, 127.4278793
37: -56.4676743, 61.1958351, -56.3186378, 61.1684761, -116.4645462, 116.3266602
38: -75.4443512, 82.2542953, -75.2427750, 82.1781616, -157.6225128, 157.4970703
39: -64.0759277, 87.6813202, -63.8657074, 87.6190262, -149.7030029, 149.5272827
40: -59.3859100, 57.7642441, -59.3106537, 57.7157936, -117.1016998, 117.0748978
41: -50.9210854, 44.2759743, -50.8556442, 44.2502899, -95.1713715, 95.1316223
42: -56.1752663, 40.2039642, -56.1350555, 40.1880684, -96.3633347, 96.3390198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706584
time: 152.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706588
time: 92.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -92.8975143, 36.9834328, -92.5545273, 36.7724991, -129.6700134, 129.5379639
1: -59.5508919, 30.0493126, -59.3740540, 29.7722759, -88.2898407, 88.4209671
2: -49.5628090, 30.9332809, -49.4428787, 30.7810326, -80.3438416, 80.3761597
3: -62.4698677, 31.3550529, -62.3808594, 31.1749229, -93.6447906, 93.7359161
4: -54.9838905, 43.8221207, -54.8279228, 43.5392990, -98.5231934, 98.6500397
5: -67.3814850, 37.0095901, -67.2722778, 36.8369026, -104.2183838, 104.2818680
6: -69.2842712, 41.5964546, -68.9932556, 41.4015694, -110.6858368, 110.5897064
7: -86.7332382, 27.3750076, -86.6179581, 27.1310978, -113.3517151, 113.4910355
8: -64.8748169, 52.0559158, -64.6999283, 51.7639542, -116.6387711, 116.7558441
9: -39.2774544, 36.0310097, -39.0915604, 35.7895012, -75.0669556, 75.1225739
10: -77.2971497, 48.0415726, -77.0839386, 47.6188736, -124.9160233, 125.1255112
11: -93.1797028, 14.5074577, -93.0609436, 14.2125473, -104.7546692, 104.9437027
12: -59.0314941, 52.1072769, -58.8808746, 52.0028343, -111.0343323, 110.9881516
13: -64.5043488, 66.8444977, -63.9878197, 66.6192017, -131.1235504, 130.8323212
14: -142.6316986, 21.3892155, -142.2535858, 20.9312706, -163.5629730, 163.6428070
15: -56.0378685, 46.9508934, -55.8695221, 46.6131325, -102.6510010, 102.8204193
16: -85.6767807, 29.8661385, -85.4456863, 29.5249825, -113.7529221, 113.8714905
17: -157.7870178, 23.2322617, -157.4665833, 22.6679497, -166.6823120, 167.0183716
18: -75.2340240, 49.0198822, -75.0681305, 48.6656685, -123.8996887, 124.0880127
19: -60.4896545, 14.8426876, -60.3536530, 14.7044344, -75.1940918, 75.1963425
20: -51.8213806, 18.9144630, -51.6904831, 18.7183990, -70.5397797, 70.6049500
21: -78.6473083, 18.8414001, -78.4928055, 18.5913124, -96.5473022, 96.6291351
22: -82.8419342, 25.0453224, -82.6588287, 24.8195953, -107.6615295, 107.7041473
23: -52.0817909, 24.6994247, -51.9402618, 24.5408478, -76.6226349, 76.6396866
24: -50.4138184, 33.2348976, -50.1296883, 33.0486717, -83.4624939, 83.3645859
25: -46.6116486, 35.9268074, -46.3426056, 35.7963409, -82.4079895, 82.2694092
26: -77.1751709, 48.0507965, -76.9745865, 47.6553345, -124.8305054, 125.0253830
27: -73.2189713, 30.6554909, -73.0733185, 30.3881588, -103.6071320, 103.7288055
28: -58.1591034, 31.3486919, -58.0469284, 31.2070999, -89.3662033, 89.3956223
29: -91.7549591, 16.1285706, -91.5372925, 15.8395176, -107.0489807, 107.1305313
30: -70.0264130, 37.4648514, -69.8496552, 37.2094917, -107.2359009, 107.3145065
31: -66.3516235, 18.8351898, -66.1670685, 18.6647873, -85.0164108, 85.0022583
32: -67.6898499, 42.3046265, -67.3677063, 42.0573235, -109.7471771, 109.6723328
33: -59.2577515, 82.4361649, -58.6543274, 82.2443848, -139.6884460, 139.2520752
34: -60.7821426, 69.1308441, -60.4414978, 68.9540634, -128.2754822, 128.0706329
35: -51.1166687, 76.7399292, -50.6105385, 76.5581589, -127.3865204, 127.0335159
36: -61.0203133, 67.1245117, -60.5356445, 66.8887939, -127.9091034, 127.6412354
37: -56.8895187, 61.2753754, -56.3356018, 61.1841202, -116.9063873, 116.4232559
38: -75.8834381, 82.5181427, -75.2543716, 82.2290421, -158.1124878, 157.7725220
39: -64.6099701, 87.8980865, -63.8825455, 87.6667862, -150.2897339, 149.7609558
40: -59.6622353, 57.9227562, -59.3260841, 57.7437477, -117.4059830, 117.2488403
41: -51.1709709, 44.4353905, -50.8621902, 44.2756310, -95.4466019, 95.2975769
42: -56.3351746, 40.3633881, -56.1407166, 40.2119217, -96.5470963, 96.5041046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033125
time: 85.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033128
time: 113.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -92.5540314, 36.7697220, -92.8186646, 36.9097595, -129.4637909, 129.5883789
1: -59.3698997, 29.7704430, -59.5424004, 29.8992157, -88.2281494, 88.3146362
2: -49.4411049, 30.7782860, -49.5731392, 30.8830185, -80.3241272, 80.3514252
3: -62.3738174, 31.1712894, -62.4377136, 31.2849388, -93.6587524, 93.6090012
4: -54.8285713, 43.5375519, -55.0070686, 43.6820602, -98.5106354, 98.5446167
5: -67.2664337, 36.8337517, -67.3741760, 36.9547920, -104.2212219, 104.2079315
6: -68.9657593, 41.4004135, -69.1184082, 41.6061783, -110.5719376, 110.5188217
7: -86.6053772, 27.1268291, -86.7810669, 27.2612667, -113.3434906, 113.4226990
8: -64.6988220, 51.7591705, -64.9214096, 51.9183884, -116.6172104, 116.6805801
9: -39.0900726, 35.7853966, -39.1912346, 35.9096451, -74.9997177, 74.9766312
10: -77.0830307, 47.6153564, -77.3357391, 47.8157463, -124.8987732, 124.9510956
11: -93.0200806, 14.2080193, -93.1959381, 14.3231297, -104.7037430, 104.7764893
12: -58.8763657, 52.0004234, -58.9898224, 52.0727310, -110.9490967, 110.9902496
13: -63.9860725, 66.6195526, -64.1765442, 66.8660889, -130.8521576, 130.7960968
14: -142.2544556, 20.9232330, -142.6544037, 21.1093216, -163.3637695, 163.5776367
15: -55.8659325, 46.6056442, -56.0566978, 46.7700806, -102.6360168, 102.6623383
16: -85.4272308, 29.5221214, -85.6497116, 29.6872711, -113.6647949, 113.7355652
17: -157.4630432, 22.6622009, -157.8498993, 22.8976936, -166.5820923, 166.8469543
18: -75.0649261, 48.6600342, -75.2429352, 48.7891121, -123.8540344, 123.9029694
19: -60.3492737, 14.7037401, -60.4584084, 14.7695675, -75.1188431, 75.1621475
20: -51.6892319, 18.7151184, -51.7857590, 18.7880516, -70.4772797, 70.5008774
21: -78.4835281, 18.5891762, -78.6300354, 18.6895752, -96.4702454, 96.5193939
22: -82.6573029, 24.7997799, -82.8503189, 24.9141922, -107.5714951, 107.6501007
23: -51.9386749, 24.5400314, -52.0352631, 24.6017952, -76.5404663, 76.5752945
24: -50.1314888, 33.0444412, -50.2800102, 33.1041641, -83.2356567, 83.3244476
25: -46.3418274, 35.7757683, -46.4855118, 35.8523483, -82.1941757, 82.2612762
26: -76.9726868, 47.6514587, -77.1427612, 47.7839737, -124.7566605, 124.7942200
27: -73.0727539, 30.3870010, -73.1930084, 30.4741688, -103.5469208, 103.5800095
28: -58.0472374, 31.2050018, -58.1275826, 31.2668571, -89.3140945, 89.3325806
29: -91.5359039, 15.8361874, -91.7551117, 15.9546757, -106.9337158, 107.0635376
30: -69.8519287, 37.2049522, -69.9722137, 37.2941551, -107.1460876, 107.1771698
31: -66.1665192, 18.6627769, -66.2735443, 18.7426376, -84.9091568, 84.9363251
32: -67.3503265, 42.0581779, -67.5225677, 42.2769547, -109.6272812, 109.5807495
33: -58.6503410, 82.2347031, -58.9079132, 82.5127487, -139.3520203, 139.3071594
34: -60.4378052, 68.9526367, -60.6054535, 69.1530151, -128.1366577, 128.0572815
35: -50.6075287, 76.5529327, -50.8199425, 76.8079834, -127.1306992, 127.0596771
36: -60.5304298, 66.8928528, -60.7458038, 67.2116013, -127.7420349, 127.6216736
37: -56.3251114, 61.1798515, -56.5848961, 61.3082008, -116.4628906, 116.5799026
38: -75.2460938, 82.2324219, -75.5272141, 82.5895844, -157.8356781, 157.7596436
39: -63.8785286, 87.6555634, -64.1846619, 88.0148010, -149.9072876, 149.8253479
40: -59.3191528, 57.7427635, -59.4897308, 57.9454575, -117.2646103, 117.2324982
41: -50.8346329, 44.2795219, -50.9940796, 44.4402351, -95.2748718, 95.2736053
42: -56.1218376, 40.2108231, -56.2326927, 40.3500633, -96.4719009, 96.4435120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753134
time: 100.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753137
time: 86.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -92.8097992, 36.9047928, -92.8186646, 36.9097595, -129.7195587, 129.7234497
1: -59.5348930, 29.8958378, -59.5424004, 29.8992157, -88.3672943, 88.4113312
2: -49.5656815, 30.8785629, -49.5731392, 30.8830185, -80.4487000, 80.4517059
3: -62.4287720, 31.2790031, -62.4377136, 31.2849388, -93.7137146, 93.7167206
4: -55.0009804, 43.6779099, -55.0070686, 43.6820602, -98.6830444, 98.6849823
5: -67.3659668, 36.9493752, -67.3741760, 36.9547920, -104.3207550, 104.3235474
6: -69.0885391, 41.5996895, -69.1184082, 41.6061783, -110.6947174, 110.7180939
7: -86.7616043, 27.2553329, -86.7810669, 27.2612667, -113.4768448, 113.5230942
8: -64.9131165, 51.9111557, -64.9214096, 51.9183884, -116.8315048, 116.8325653
9: -39.1877975, 35.9038086, -39.1912346, 35.9096451, -75.0974426, 75.0950470
10: -77.3296051, 47.8094292, -77.3357391, 47.8157463, -125.1453552, 125.1451721
11: -93.1510925, 14.3172092, -93.1959381, 14.3231297, -104.8038940, 104.8540421
12: -58.9828300, 52.0667076, -58.9898224, 52.0727310, -111.0555573, 111.0565338
13: -64.1724930, 66.8588104, -64.1765442, 66.8660889, -131.0385742, 131.0353546
14: -142.6444092, 21.1000099, -142.6544037, 21.1093216, -163.7537231, 163.7544098
15: -56.0498466, 46.7604256, -56.0566978, 46.7700806, -102.8199310, 102.8171234
16: -85.6267395, 29.6824493, -85.6497116, 29.6872711, -113.8334045, 113.8607483
17: -157.8382263, 22.8894005, -157.8498993, 22.8976936, -166.8963776, 166.9966736
18: -75.2334747, 48.7817345, -75.2429352, 48.7891121, -124.0225830, 124.0246735
19: -60.4509583, 14.7674961, -60.4584084, 14.7695675, -75.2205276, 75.2259064
20: -51.7812309, 18.7835999, -51.7857590, 18.7880516, -70.5692825, 70.5693588
21: -78.6160278, 18.6861496, -78.6300354, 18.6895752, -96.5886459, 96.5987701
22: -82.8428192, 24.8928432, -82.8503189, 24.9141922, -107.7570114, 107.7431641
23: -52.0301743, 24.5996075, -52.0352631, 24.6017952, -76.6319733, 76.6348724
24: -50.2752991, 33.0991058, -50.2800102, 33.1041641, -83.3794632, 83.3791199
25: -46.4810905, 35.8306122, -46.4855118, 35.8523483, -82.3334351, 82.3161240
26: -77.1342850, 47.7779808, -77.1427612, 47.7839737, -124.9182587, 124.9207458
27: -73.1859283, 30.4716263, -73.1930084, 30.4741688, -103.6600952, 103.6646347
28: -58.1240578, 31.2632675, -58.1275826, 31.2668571, -89.3909149, 89.3908539
29: -91.7475967, 15.9498997, -91.7551117, 15.9546757, -107.1101532, 107.1324463
30: -69.9669571, 37.2880859, -69.9722137, 37.2941551, -107.2611084, 107.2602997
31: -66.2698975, 18.7392483, -66.2735443, 18.7426376, -85.0125351, 85.0127945
32: -67.5021210, 42.2734451, -67.5225677, 42.2769547, -109.7790756, 109.7960129
33: -58.9007263, 82.4976501, -58.9079132, 82.5127487, -139.5850677, 139.5544586
34: -60.5984230, 69.1483765, -60.6054535, 69.1530151, -128.2703857, 128.2298737
35: -50.8141174, 76.7979279, -50.8199425, 76.8079834, -127.3136520, 127.2833710
36: -60.7385712, 67.2091827, -60.7458038, 67.2116013, -127.9436417, 127.9233398
37: -56.5703392, 61.3011169, -56.5848961, 61.3082008, -116.6986313, 116.6911240
38: -75.5163574, 82.5847321, -75.5272141, 82.5895844, -158.1059418, 158.1119385
39: -64.1767578, 87.9960938, -64.1846619, 88.0148010, -150.1735535, 150.1370850
40: -59.4787521, 57.9407349, -59.4897308, 57.9454575, -117.4242096, 117.4304657
41: -50.9644241, 44.4376183, -50.9940796, 44.4402351, -95.4046631, 95.4317017
42: -56.2122345, 40.3444672, -56.2326927, 40.3500633, -96.5623016, 96.5771637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374431
time: 89.36 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374435
time: 94.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 186.82 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5157106, upper bound: 89.4927030
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5157106, upper bound: 89.4927030
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5003211, upper bound: 89.5254798
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5003211, upper bound: 89.5254798
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5052296, upper bound: 89.4757030
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5382481, upper bound: 89.4603084
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5052296, upper bound: 89.5378726
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.5382481, upper bound: 89.5236112
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706584
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706588
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033125
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033128
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753134
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753137
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374431
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 186.82
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374435

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -92.4013290, 36.6787148, -92.2849121, 36.5807419, -128.9820709, 128.9636230
1: -59.2609863, 29.6900768, -59.1860199, 29.5901909, -87.8423767, 87.8615417
2: -49.3257294, 30.7026215, -49.2625084, 30.6366844, -79.9624176, 79.9651337
3: -62.2291260, 31.0625229, -62.1993942, 31.0069046, -93.2360306, 93.2619171
4: -54.7626495, 43.5131073, -54.6915703, 43.4164314, -98.1790771, 98.2046814
5: -67.1529312, 36.7570648, -67.0974197, 36.6965752, -103.8495026, 103.8544846
6: -68.9726410, 41.3425560, -68.8804626, 41.3103142, -110.2829590, 110.2230225
7: -86.4493561, 27.0071297, -86.3782806, 26.9157982, -112.8689117, 112.8844299
8: -64.5051727, 51.6307602, -64.4217606, 51.5295639, -116.0347366, 116.0525208
9: -39.0636597, 35.8058167, -39.0142899, 35.7163353, -74.7799988, 74.8201065
10: -76.9883728, 47.6552277, -76.9293518, 47.5112076, -124.4995804, 124.5845795
11: -92.8527527, 14.1990910, -92.8215637, 14.1145306, -104.3310623, 104.3922195
12: -58.7826385, 51.8750763, -58.7403526, 51.8488083, -110.6314468, 110.6154327
13: -64.0633240, 66.5063019, -63.9190521, 66.4714890, -130.5348206, 130.4253540
14: -141.9286499, 20.9154778, -141.8728790, 20.7589340, -162.6875916, 162.7883606
15: -55.7798538, 46.6395569, -55.7179985, 46.5230179, -102.3028717, 102.3575592
16: -85.3631134, 29.5260277, -85.2925568, 29.4121857, -113.3408432, 113.3817444
17: -157.3063354, 22.7007141, -157.2239227, 22.5222015, -166.1335144, 166.2257843
18: -74.8594971, 48.5612450, -74.8614349, 48.4675636, -123.3270569, 123.4226837
19: -60.2966309, 14.6917191, -60.2682076, 14.6508017, -74.9474335, 74.9599304
20: -51.6216888, 18.7001400, -51.5845795, 18.6514740, -70.2731628, 70.2847214
21: -78.3734894, 18.6084747, -78.3617249, 18.5304489, -96.2063904, 96.2700195
22: -82.4815521, 24.7508736, -82.4884186, 24.6807671, -107.1623230, 107.2392883
23: -51.8812714, 24.5352707, -51.8566475, 24.4960632, -76.3773346, 76.3919220
24: -49.9856758, 32.9962082, -49.9828720, 32.9557724, -82.9414520, 82.9790802
25: -46.2568169, 35.6403885, -46.1849022, 35.6030502, -81.8598633, 81.8252869
26: -76.6974182, 47.6118088, -76.7080765, 47.5157394, -124.2131577, 124.3198853
27: -72.9257050, 30.4066849, -72.9543610, 30.3430309, -103.2687378, 103.3610458
28: -57.9614830, 31.1818905, -57.9473877, 31.1490822, -89.1105652, 89.1292801
29: -91.3849487, 15.8570080, -91.3725128, 15.7655239, -106.5944901, 106.6740265
30: -69.6883316, 37.1611824, -69.7069244, 37.1045456, -106.7928772, 106.8681030
31: -66.0914383, 18.6316509, -66.0429459, 18.5786362, -84.6700745, 84.6745987
32: -67.3595886, 42.0293770, -67.2723236, 41.9652328, -109.3248215, 109.3016968
33: -58.6700401, 82.0454559, -58.4794769, 81.9797821, -138.8245392, 138.6980743
34: -60.4027557, 68.8467255, -60.2924194, 68.7714844, -127.6920166, 127.6595993
35: -50.6773796, 76.4942169, -50.5163307, 76.4254684, -126.8435287, 126.7496948
36: -60.5446892, 66.8363495, -60.3782349, 66.7590485, -127.2976074, 127.2082062
37: -56.2564468, 61.0308304, -56.0929337, 60.9963989, -116.0730896, 115.9427872
38: -75.3287201, 82.1918259, -75.1162567, 82.1109314, -157.4396515, 157.3080750
39: -63.9824142, 87.5676727, -63.7644196, 87.4866333, -149.4956512, 149.3565369
40: -59.2642784, 57.6835480, -59.1781387, 57.6303368, -116.8946152, 116.8616867
41: -50.8377991, 44.2387848, -50.7426987, 44.2105598, -95.0483551, 94.9814835
42: -56.1146774, 40.1460228, -56.0540161, 40.1245689, -96.2392426, 96.2000427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4797620
time: 110.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4896864
time: 137.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -92.4013290, 36.6787148, -92.5253296, 36.7598801, -129.1612091, 129.2040405
1: -59.2609863, 29.6900768, -59.3523140, 29.7638283, -88.0236359, 88.0359650
2: -49.3257294, 30.7026215, -49.4307098, 30.7692776, -80.0950089, 80.1333313
3: -62.2291260, 31.0625229, -62.3680267, 31.1575928, -93.3867188, 93.4305496
4: -54.7626495, 43.5131073, -54.8079681, 43.5258675, -98.2885132, 98.3210754
5: -67.1529312, 36.7570648, -67.2593689, 36.8191223, -103.9720535, 104.0164337
6: -68.9726410, 41.3425560, -68.9577942, 41.3609047, -110.3335419, 110.3003540
7: -86.4493561, 27.0071297, -86.5898285, 27.1167068, -113.0694199, 113.1050034
8: -64.5051727, 51.6307602, -64.6731644, 51.7460251, -116.2511978, 116.3039246
9: -39.0636597, 35.8058167, -39.0793266, 35.7780685, -74.8417282, 74.8851471
10: -76.9883728, 47.6552277, -77.0416336, 47.6022186, -124.5905914, 124.6968613
11: -92.8527527, 14.1990910, -92.9959412, 14.2017784, -104.4165039, 104.5643311
12: -58.7826385, 51.8750763, -58.8596992, 51.9905930, -110.7732315, 110.7347717
13: -64.0633240, 66.5063019, -63.9758263, 66.5709000, -130.6342163, 130.4821320
14: -141.9286499, 20.9154778, -142.1746063, 20.9185257, -162.8471680, 163.0900879
15: -55.7798538, 46.6395569, -55.8398476, 46.5957184, -102.3755722, 102.4794006
16: -85.3631134, 29.5260277, -85.3989334, 29.5100746, -113.4380341, 113.4786453
17: -157.3063354, 22.7007141, -157.3970032, 22.6490173, -166.2659454, 166.4069214
18: -74.8594971, 48.5612450, -75.0212250, 48.6533279, -123.5128250, 123.5824738
19: -60.2966309, 14.6917191, -60.3308640, 14.6974039, -74.9940338, 75.0225830
20: -51.6216888, 18.7001400, -51.6744270, 18.7102985, -70.3319855, 70.3745651
21: -78.3734894, 18.6084747, -78.4514008, 18.5848160, -96.2591553, 96.3541336
22: -82.4815521, 24.7508736, -82.6073914, 24.7940559, -107.2756042, 107.3582611
23: -51.8812714, 24.5352707, -51.9210243, 24.5351677, -76.4164429, 76.4562988
24: -49.9856758, 32.9962082, -50.0904007, 33.0408745, -83.0265503, 83.0866089
25: -46.2568169, 35.6403885, -46.3237724, 35.7715950, -82.0284119, 81.9641571
26: -76.6974182, 47.6118088, -76.9258423, 47.6438065, -124.3412247, 124.5376511
27: -72.9257050, 30.4066849, -73.0324249, 30.3820114, -103.3077164, 103.4391098
28: -57.9614830, 31.1818905, -58.0300980, 31.1989727, -89.1604538, 89.2119904
29: -91.3849487, 15.8570080, -91.4857559, 15.8304291, -106.6617279, 106.7909546
30: -69.6883316, 37.1611824, -69.8134460, 37.1985321, -106.8868637, 106.9746246
31: -66.0914383, 18.6316509, -66.1521759, 18.6568584, -84.7482986, 84.7838287
32: -67.3595886, 42.0293770, -67.3327637, 42.0223045, -109.3818970, 109.3621368
33: -58.6700401, 82.0454559, -58.6357536, 82.1922760, -139.0434875, 138.8611145
34: -60.4027557, 68.8467255, -60.4247971, 68.9289856, -127.8616409, 127.7966919
35: -50.6773796, 76.4942169, -50.5958405, 76.5162354, -126.9095001, 126.8185959
36: -60.5446892, 66.8363495, -60.5196609, 66.8420410, -127.3867340, 127.3532715
37: -56.2564468, 61.0308304, -56.3041000, 61.1613846, -116.2443237, 116.1571960
38: -75.3287201, 82.1918259, -75.2318878, 82.1733398, -157.5020599, 157.4237061
39: -63.9824142, 87.5676727, -63.8578300, 87.6002731, -149.5919189, 149.4445953
40: -59.2642784, 57.6835480, -59.2997055, 57.7110786, -116.9753571, 116.9832535
41: -50.8377991, 44.2387848, -50.8259773, 44.2476578, -95.0854568, 95.0647583
42: -56.1146774, 40.1460228, -56.1146126, 40.1824799, -96.2971573, 96.2606354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4797620
time: 99.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4896864
time: 121.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -92.6575089, 36.8042336, -92.3052444, 36.5883865, -129.2458954, 129.1094818
1: -59.3847809, 29.8756886, -59.2002640, 29.5952740, -87.9744949, 88.0613556
2: -49.3946915, 30.8005905, -49.2672081, 30.6439762, -80.0386658, 80.0677948
3: -62.3012733, 31.2041397, -62.2032585, 31.0183048, -93.3195801, 93.4073944
4: -54.8675385, 43.7125854, -54.7054367, 43.4257164, -98.2932587, 98.4180222
5: -67.2195358, 36.8869019, -67.1021500, 36.7089195, -103.9284515, 103.9890518
6: -69.2068481, 41.5458031, -68.8860626, 41.3445587, -110.5514069, 110.4318695
7: -86.5217590, 27.1738567, -86.3869171, 26.9241886, -112.9583435, 113.0512543
8: -64.6235046, 51.8392410, -64.4402390, 51.5402527, -116.1637573, 116.2794800
9: -39.2124672, 35.9692535, -39.0230942, 35.7219353, -74.9344025, 74.9923477
10: -77.1849518, 47.9505310, -76.9655151, 47.5215111, -124.7064667, 124.9160461
11: -93.0054016, 14.4202023, -92.8416901, 14.1194057, -104.5023346, 104.6410675
12: -58.9120827, 51.9655266, -58.7545433, 51.8549919, -110.7670746, 110.7200699
13: -64.4477997, 66.7448502, -63.9269562, 66.5125122, -130.9603119, 130.6718140
14: -142.3302002, 21.2296925, -141.9418945, 20.7623329, -163.0925293, 163.1715851
15: -55.9157791, 46.8780632, -55.7408447, 46.5307770, -102.4465561, 102.6189117
16: -85.5704727, 29.7682114, -85.3163300, 29.4222660, -113.5597305, 113.6509018
17: -157.6139832, 23.1053810, -157.2818298, 22.5328445, -166.4584961, 166.6940002
18: -75.0742493, 48.8341522, -74.8988495, 48.4725037, -123.5467529, 123.7330017
19: -60.4271278, 14.7960815, -60.2835388, 14.6557512, -75.0828781, 75.0796204
20: -51.7314835, 18.8558064, -51.5960960, 18.6551151, -70.3865967, 70.4519043
21: -78.5576324, 18.7869797, -78.3890915, 18.5335083, -96.4050446, 96.4766617
22: -82.7228699, 24.9320984, -82.5323944, 24.6849613, -107.4078293, 107.4644928
23: -52.0172577, 24.6603031, -51.8707733, 24.4995403, -76.5167999, 76.5310745
24: -50.3058624, 33.1498260, -50.0174713, 32.9585114, -83.2643738, 83.1672974
25: -46.4722824, 35.7582817, -46.1992836, 35.6060562, -82.0783386, 81.9575653
26: -76.9573364, 47.9228783, -76.7483826, 47.5212479, -124.4785843, 124.6712646
27: -73.1408081, 30.6165886, -72.9881897, 30.3466320, -103.4874420, 103.6047821
28: -58.0763359, 31.2988625, -57.9606972, 31.1536140, -89.2299500, 89.2595596
29: -91.6418610, 16.0636578, -91.4165497, 15.7698212, -106.8725891, 106.9321442
30: -69.9196701, 37.3709373, -69.7379150, 37.1094093, -107.0290833, 107.1088562
31: -66.2420654, 18.7569237, -66.0541840, 18.5831642, -84.8252258, 84.8111115
32: -67.6293182, 42.2479706, -67.2867889, 41.9967537, -109.6260681, 109.5347595
33: -59.1013985, 82.2237320, -58.4908752, 82.0167999, -139.2946167, 138.8890686
34: -60.6497383, 68.9734344, -60.3020782, 68.7919159, -127.9625244, 127.7982178
35: -51.0369263, 76.6492538, -50.5252075, 76.4573669, -127.2364807, 126.9140015
36: -60.8788223, 67.0415955, -60.3870010, 66.8033752, -127.6772461, 127.4215851
37: -56.6781082, 61.1104317, -56.1099319, 61.0120430, -116.5147552, 116.0394287
38: -75.7676697, 82.4557190, -75.1278687, 82.1618042, -157.9294739, 157.5835876
39: -64.5160370, 87.7844772, -63.7812347, 87.5344086, -150.0820312, 149.5903015
40: -59.5406837, 57.8421516, -59.1935577, 57.6582718, -117.1989594, 117.0357056
41: -51.0876656, 44.3980789, -50.7492523, 44.2359314, -95.3235931, 95.1473312
42: -56.2746277, 40.3054314, -56.0596924, 40.1484528, -96.4230804, 96.3651276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 602

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4362198, upper bound: 89.5144937
time: 98.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4929167, upper bound: 89.5229147
time: 105.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -92.6575089, 36.8042336, -92.5456543, 36.7675400, -129.4250488, 129.3498840
1: -59.3847809, 29.8756886, -59.3665543, 29.7689095, -88.1557617, 88.2357483
2: -49.3946915, 30.8005905, -49.4354134, 30.7765846, -80.1712799, 80.2360077
3: -62.3012733, 31.2041397, -62.3718987, 31.1690063, -93.4702759, 93.5760345
4: -54.8675385, 43.7125854, -54.8218231, 43.5351448, -98.4026794, 98.5344086
5: -67.2195358, 36.8869019, -67.2640839, 36.8314857, -104.0510254, 104.1509857
6: -69.2068481, 41.5458031, -68.9633942, 41.3951263, -110.6019745, 110.5092010
7: -86.5217590, 27.1738567, -86.5984497, 27.1251354, -113.1588745, 113.2718201
8: -64.6235046, 51.8392410, -64.6916504, 51.7567368, -116.3802414, 116.5308914
9: -39.2124672, 35.9692535, -39.0881348, 35.7836761, -74.9961395, 75.0573883
10: -77.1849518, 47.9505310, -77.0777969, 47.6125526, -124.7975006, 125.0283279
11: -93.0054016, 14.4202023, -93.0160675, 14.2066498, -104.5877686, 104.8131943
12: -58.9120827, 51.9655266, -58.8738670, 51.9968185, -110.9089050, 110.8393936
13: -64.4477997, 66.7448502, -63.9837608, 66.6119537, -131.0597534, 130.7286072
14: -142.3302002, 21.2296925, -142.2436066, 20.9219284, -163.2521362, 163.4732971
15: -55.9157791, 46.8780632, -55.8626900, 46.6035233, -102.5193024, 102.7407532
16: -85.5704727, 29.7682114, -85.4227142, 29.5201721, -113.6569366, 113.7477722
17: -157.6139832, 23.1053810, -157.4549255, 22.6596718, -166.5908813, 166.8750916
18: -75.0742493, 48.8341522, -75.0586624, 48.6582718, -123.7325211, 123.8928146
19: -60.4271278, 14.7960815, -60.3462181, 14.7023735, -75.1295013, 75.1423035
20: -51.7314835, 18.8558064, -51.6859474, 18.7139454, -70.4454269, 70.5417557
21: -78.5576324, 18.7869797, -78.4787598, 18.5878944, -96.4577789, 96.5607910
22: -82.7228699, 24.9320984, -82.6513367, 24.7982426, -107.5211105, 107.5834351
23: -52.0172577, 24.6603031, -51.9351578, 24.5386562, -76.5559158, 76.5954590
24: -50.3058624, 33.1498260, -50.1249580, 33.0436020, -83.3494644, 83.2747803
25: -46.4722824, 35.7582817, -46.3381805, 35.7746201, -82.2469025, 82.0964661
26: -76.9573364, 47.9228783, -76.9661331, 47.6493263, -124.6066589, 124.8890076
27: -73.1408081, 30.6165886, -73.0662384, 30.3856030, -103.5264130, 103.6828308
28: -58.0763359, 31.2988625, -58.0434036, 31.2035160, -89.2798538, 89.3422699
29: -91.6418610, 16.0636578, -91.5297852, 15.8347340, -106.9398804, 107.0490494
30: -69.9196701, 37.3709373, -69.8444061, 37.2034187, -107.1230927, 107.2153473
31: -66.2420654, 18.7569237, -66.1634369, 18.6613865, -84.9034500, 84.9203644
32: -67.6293182, 42.2479706, -67.3472443, 42.0538101, -109.6831284, 109.5952148
33: -59.1013985, 82.2237320, -58.6471176, 82.2292938, -139.5135498, 139.0521088
34: -60.6497383, 68.9734344, -60.4344559, 68.9494324, -128.1321411, 127.9353561
35: -51.0369263, 76.6492538, -50.6047058, 76.5481110, -127.3024673, 126.9829025
36: -60.8788223, 67.0415955, -60.5284271, 66.8863678, -127.7651901, 127.5666275
37: -56.6781082, 61.1104317, -56.3210449, 61.1770325, -116.6859741, 116.2538605
38: -75.7676697, 82.4557190, -75.2434845, 82.2241974, -157.9918671, 157.6992035
39: -64.5160370, 87.7844772, -63.8746567, 87.6480408, -150.1783447, 149.6783447
40: -59.5406837, 57.8421516, -59.3151398, 57.7390251, -117.2797089, 117.1572876
41: -51.0876656, 44.3980789, -50.8325157, 44.2730217, -95.3606873, 95.2305908
42: -56.2746277, 40.3054314, -56.1202698, 40.2063293, -96.4809570, 96.4257050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 602

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4362198, upper bound: 89.5144937
time: 108.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4929167, upper bound: 89.5229147
time: 61.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 171.94 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4797620
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4896864
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4797620
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4472942, upper bound: 89.4896864
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4362198, upper bound: 89.5144937
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4929167, upper bound: 89.5229147
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4362198, upper bound: 89.5144937
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 171.94
Output dim: 37, lower bound: -89.4929167, upper bound: 89.5229147
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.5052296, upper bound: 89.4757030
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.5382481, upper bound: 89.4603084
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.5052296, upper bound: 89.5378726
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.5382481, upper bound: 89.5236112
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706584
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4757030, upper bound: 89.5706588
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033125
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4603084, upper bound: 89.6033128
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753134
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4973256, upper bound: 89.5753137
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374431
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 171.94
Output dim: 37, lower bound: -89.4973256, upper bound: 89.6374435
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=116.73698425292969
rel_dist={37: [-89.67935830190423, 89.67935830286939]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 729

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.2327703, upper bound: 88.1608162
time: 123.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.2327703, upper bound: 88.2327702
time: 86.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 209.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 209.48
Output dim: 37, lower bound: -88.2327703, upper bound: 88.1608162
IS_A2, status: Status.UNKNOWN, split count: 1, time: 209.48
Output dim: 37, lower bound: -88.2327703, upper bound: 88.2327702

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -92.5766144, 36.7272797, -92.7449646, 36.8980675, -129.4746857, 129.4722443
1: -59.3741531, 29.7235413, -59.4890823, 29.8866119, -87.6826935, 87.6273041
2: -49.3999863, 30.7474213, -49.5176086, 30.8731804, -80.2731628, 80.2650299
3: -62.2667351, 31.1304169, -62.3827667, 31.2714767, -93.5382080, 93.5131836
4: -54.8909836, 43.5707092, -54.9741478, 43.6724968, -98.5634766, 98.5448608
5: -67.2087097, 36.8288574, -67.3224030, 36.9449310, -104.1536407, 104.1512604
6: -69.0129242, 41.5568008, -69.0895844, 41.5921059, -110.6050262, 110.6463852
7: -86.5544739, 27.0560989, -86.7059174, 27.2446899, -112.7651596, 112.7307587
8: -64.6697540, 51.6969833, -64.8413773, 51.9033127, -116.5730667, 116.5383606
9: -39.1258163, 35.8437691, -39.1754990, 35.8996811, -75.0254974, 75.0192719
10: -77.2260742, 47.7212410, -77.3113785, 47.8020782, -125.0281525, 125.0326233
11: -92.9816284, 14.2313957, -93.1346664, 14.3025932, -103.6325073, 103.7157364
12: -58.8658867, 51.9315338, -58.9773865, 52.0345154, -110.9004059, 110.9089203
13: -64.1183701, 66.7683258, -64.1644592, 66.8542404, -130.9726105, 130.9327850
14: -142.3578949, 20.9416828, -142.5726929, 21.0913696, -163.4492645, 163.5143738
15: -55.9344368, 46.6895332, -56.0291405, 46.7518387, -102.6862793, 102.7186737
16: -85.5276489, 29.5867386, -85.6190491, 29.6719570, -113.0411453, 113.0455017
17: -157.6787872, 22.7648430, -157.8075256, 22.8796387, -165.2660980, 165.2801666
18: -75.0799866, 48.5977325, -75.2276001, 48.7274475, -123.8074341, 123.8253326
19: -60.3920822, 14.7219152, -60.4465179, 14.7638493, -75.1559296, 75.1684341
20: -51.6937790, 18.7260647, -51.7706261, 18.7733994, -70.4671783, 70.4966888
21: -78.5317917, 18.6328545, -78.6076660, 18.6783752, -96.0174484, 96.0490646
22: -82.7314148, 24.7807159, -82.8318558, 24.8702202, -107.6016388, 107.6125717
23: -51.9689713, 24.5617599, -52.0247726, 24.5928574, -76.5618286, 76.5865326
24: -50.1737747, 33.0148849, -50.2709045, 33.0765495, -83.2503204, 83.2857895
25: -46.3452454, 35.6632195, -46.4748230, 35.7916565, -82.1369019, 82.1380463
26: -76.9218979, 47.6517982, -77.1185760, 47.7429161, -124.6648102, 124.7703705
27: -73.1140900, 30.4339638, -73.1779327, 30.4661293, -103.5802155, 103.6118927
28: -58.0445290, 31.2149391, -58.1186829, 31.2556705, -89.3002014, 89.3336182
29: -91.6434479, 15.8860188, -91.7308197, 15.9432087, -106.1077271, 106.1453705
30: -69.8655548, 37.1955566, -69.9610748, 37.2683983, -107.1339569, 107.1566315
31: -66.1632385, 18.6622314, -66.2622833, 18.7315025, -84.8947449, 84.9245148
32: -67.4447632, 42.2245445, -67.5047760, 42.2681351, -109.7128983, 109.7293243
33: -58.7477188, 82.2953339, -58.8927994, 82.4464569, -139.0039978, 139.0026550
34: -60.4687729, 68.9983368, -60.5876656, 69.1066132, -127.5144424, 127.5209122
35: -50.7373390, 76.7162170, -50.8071671, 76.7785034, -126.7662964, 126.7712097
36: -60.5996056, 67.1382904, -60.7290955, 67.1954193, -127.4513474, 127.5214615
37: -56.3634644, 61.1411667, -56.5573463, 61.2553558, -116.2139435, 116.2940521
38: -75.4035110, 82.5364304, -75.5031509, 82.5834808, -157.8482666, 157.9067230
39: -64.0875854, 87.8962097, -64.1699677, 87.9815826, -149.4230347, 149.4216919
40: -59.3607330, 57.8673897, -59.4678955, 57.9274635, -117.2881927, 117.3352814
41: -50.8831215, 44.4067078, -50.9581871, 44.4319839, -95.3151093, 95.3648987
42: -56.1530457, 40.2920837, -56.2103424, 40.3376923, -96.4907379, 96.5024261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=490, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1575662, upper bound: 88.1407953
time: 90.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1575662, upper bound: 88.1382357
time: 82.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -92.8168945, 36.9064636, -92.8247910, 36.9108696, -129.7277679, 129.7312622
1: -59.5403709, 29.8971672, -59.5470810, 29.9001579, -87.8243027, 87.8699493
2: -49.5681419, 30.8800659, -49.5748100, 30.8840237, -80.4521637, 80.4548798
3: -62.4353523, 31.2812347, -62.4433632, 31.2865219, -93.7218781, 93.7245941
4: -55.0073509, 43.6801758, -55.0127831, 43.6838455, -98.6911926, 98.6929626
5: -67.3706284, 36.9514656, -67.3778687, 36.9562988, -104.3269272, 104.3293304
6: -69.0903168, 41.6073341, -69.1169586, 41.6130905, -110.7034073, 110.7242889
7: -86.7659836, 27.2570705, -86.7833786, 27.2623577, -112.9671249, 113.0135040
8: -64.9211044, 51.9135361, -64.9285126, 51.9199677, -116.8410721, 116.8420486
9: -39.1908340, 35.9055328, -39.1938934, 35.9107132, -75.1015472, 75.0994263
10: -77.3383026, 47.8122673, -77.3437881, 47.8178902, -125.1561890, 125.1560516
11: -93.1560059, 14.3186169, -93.1961365, 14.3238888, -103.8125000, 103.8613281
12: -58.9852676, 52.0732651, -58.9914970, 52.0786552, -111.0639191, 111.0647583
13: -64.1752014, 66.8678436, -64.1787872, 66.8743286, -131.0495300, 131.0466309
14: -142.6594696, 21.1012630, -142.6683044, 21.1095886, -163.7690582, 163.7695618
15: -56.0562820, 46.7623138, -56.0623512, 46.7709045, -102.8271866, 102.8246613
16: -85.6339722, 29.6846275, -85.6545410, 29.6888752, -113.1471100, 113.1740723
17: -157.8517761, 22.8917389, -157.8622131, 22.8991051, -165.3658600, 165.4704590
18: -75.2397614, 48.7835312, -75.2482224, 48.7900047, -124.0297699, 124.0317535
19: -60.4546967, 14.7685261, -60.4613152, 14.7703686, -75.2250671, 75.2298431
20: -51.7836494, 18.7849102, -51.7876854, 18.7888680, -70.5725174, 70.5725937
21: -78.6214066, 18.6872406, -78.6339569, 18.6902885, -96.1142883, 96.1245728
22: -82.8503723, 24.8939972, -82.8570099, 24.9130821, -107.7634583, 107.7510071
23: -52.0333862, 24.6008911, -52.0379028, 24.6028481, -76.6362305, 76.6387939
24: -50.2813492, 33.0999718, -50.2855492, 33.1044693, -83.3858185, 83.3855209
25: -46.4842339, 35.8317833, -46.4881935, 35.8507919, -82.3350220, 82.3199768
26: -77.1396332, 47.7798233, -77.1471558, 47.7851791, -124.9248123, 124.9269791
27: -73.1921539, 30.4729614, -73.1984634, 30.4752312, -103.6673889, 103.6714249
28: -58.1272621, 31.2648582, -58.1304245, 31.2680492, -89.3953094, 89.3952789
29: -91.7566528, 15.9509907, -91.7633209, 15.9552364, -106.2227936, 106.2465057
30: -69.9721069, 37.2895775, -69.9767609, 37.2949982, -107.2671051, 107.2663422
31: -66.2725830, 18.7404671, -66.2758408, 18.7434692, -85.0160522, 85.0163116
32: -67.5052643, 42.2814713, -67.5235214, 42.2845993, -109.7898636, 109.8049927
33: -58.9039345, 82.5077591, -58.9103165, 82.5211182, -139.2436829, 139.2126770
34: -60.6011429, 69.1558380, -60.6074028, 69.1598663, -127.7089996, 127.6667023
35: -50.8169403, 76.8069000, -50.8221512, 76.8156433, -126.8609772, 126.8309708
36: -60.7409821, 67.2212677, -60.7474213, 67.2233887, -127.6274033, 127.6056519
37: -56.5746498, 61.3061066, -56.5875969, 61.3123703, -116.4869690, 116.4780121
38: -75.5191956, 82.5988159, -75.5289001, 82.6031418, -157.9833221, 157.9646301
39: -64.1811829, 88.0098114, -64.1881943, 88.0265884, -149.5497437, 149.5129700
40: -59.4822884, 57.9480896, -59.4920311, 57.9523087, -117.4346008, 117.4401245
41: -50.9663925, 44.4438477, -50.9928780, 44.4461861, -95.4125824, 95.4367218
42: -56.2136574, 40.3499680, -56.2319527, 40.3549385, -96.5685959, 96.5819244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=439, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1598

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.1575662, upper bound: 88.2127357
time: 91.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.1575662, upper bound: 88.2101554
time: 101.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 195.68 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 195.68
Output dim: 37, lower bound: -88.1575662, upper bound: 88.1407953
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 195.68
Output dim: 37, lower bound: -88.1575662, upper bound: 88.1382357
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 195.68
Output dim: 37, lower bound: -88.1575662, upper bound: 88.2127357
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 195.68
Output dim: 37, lower bound: -88.1575662, upper bound: 88.2101554

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -92.7433090, 36.8948708, -92.5619583, 36.7741165, -129.5174255, 129.4568329
1: -59.4885445, 29.8884144, -59.3766174, 29.7734451, -87.6458893, 87.6892853
2: -49.5283203, 30.8694439, -49.4477806, 30.7822762, -80.3105927, 80.3172226
3: -62.4181213, 31.2651672, -62.3818169, 31.1765404, -93.5946655, 93.6469879
4: -54.9503784, 43.6664467, -54.8340263, 43.5412292, -98.4916077, 98.5004730
5: -67.3380737, 36.9353485, -67.2736816, 36.8385773, -104.1766510, 104.2090302
6: -69.0800476, 41.5418282, -68.9924011, 41.4061852, -110.4862366, 110.5342255
7: -86.7136383, 27.2464962, -86.6227646, 27.1321201, -112.7819519, 112.8375244
8: -64.8476257, 51.8968277, -64.7062073, 51.7656364, -116.6132660, 116.6030350
9: -39.1679001, 35.8948174, -39.0931473, 35.7905846, -74.9584808, 74.9879608
10: -77.2541809, 47.7956963, -77.0885010, 47.6210098, -124.8751907, 124.8842010
11: -93.1173935, 14.3098221, -93.0602417, 14.2132893, -103.6589737, 103.7104340
12: -58.9654655, 52.0527649, -58.8825798, 52.0057983, -110.9712677, 110.9353485
13: -64.1583405, 66.7890930, -63.9897079, 66.6259766, -130.7843170, 130.7788086
14: -142.5306396, 21.0923634, -142.2633057, 20.9315624, -163.4622040, 163.3556671
15: -55.9957848, 46.7494278, -55.8719940, 46.6142349, -102.6100159, 102.6214218
16: -85.5727997, 29.6695061, -85.4477997, 29.5264187, -112.9221878, 112.9481812
17: -157.7247620, 22.8777046, -157.4734650, 22.6695862, -165.0065918, 165.0513306
18: -75.1859741, 48.7717018, -75.0733948, 48.6665077, -123.8524780, 123.8450928
19: -60.4287987, 14.7623558, -60.3558769, 14.7055817, -75.1343842, 75.1182327
20: -51.7673531, 18.7768059, -51.6932564, 18.7190685, -70.4864197, 70.4700623
21: -78.5844421, 18.6788826, -78.4960785, 18.5922127, -95.9823837, 95.9754562
22: -82.7913742, 24.8854580, -82.6639709, 24.8188438, -107.6102142, 107.5494308
23: -52.0123672, 24.5923367, -51.9432220, 24.5419769, -76.5543442, 76.5355606
24: -50.2528229, 33.0947342, -50.1356773, 33.0489578, -83.3017807, 83.2304077
25: -46.4639740, 35.8243141, -46.3457947, 35.7947693, -82.2587433, 82.1701050
26: -77.0961075, 47.7684784, -76.9802094, 47.6568375, -124.7529449, 124.7486877
27: -73.1597824, 30.4641609, -73.0790482, 30.3892918, -103.5490723, 103.5432129
28: -58.1088181, 31.2539673, -58.0503922, 31.2081928, -89.3170090, 89.3043594
29: -91.6883087, 15.9435482, -91.5425873, 15.8404732, -106.0388718, 106.0084381
30: -69.9431610, 37.2789993, -69.8566132, 37.2103691, -107.1535339, 107.1356125
31: -66.2547073, 18.7319584, -66.1697845, 18.6657925, -84.9205017, 84.9017410
32: -67.4871902, 42.2138863, -67.3686066, 42.0613022, -109.5484924, 109.5824890
33: -58.8810959, 82.4139023, -58.6567268, 82.2480698, -138.9455109, 138.8642273
34: -60.5837555, 69.0885468, -60.4440536, 68.9566803, -127.4819794, 127.4329071
35: -50.8006325, 76.7189789, -50.6127167, 76.5616531, -126.5888901, 126.5335007
36: -60.7266006, 67.1065063, -60.5368462, 66.8949738, -127.2829895, 127.2801208
37: -56.5470200, 61.2632103, -56.3380585, 61.1861000, -116.3319550, 116.1837234
38: -75.4975357, 82.4728088, -75.2558289, 82.2367401, -157.5891113, 157.5643768
39: -64.1541214, 87.8842468, -63.8855286, 87.6723480, -149.1653748, 149.0837402
40: -59.4596596, 57.8809891, -59.3288994, 57.7469902, -117.2066498, 117.2098846
41: -50.9533310, 44.3922997, -50.8610916, 44.2818642, -95.2351990, 95.2533875
42: -56.2025986, 40.3091125, -56.1401443, 40.2158127, -96.4184113, 96.4492569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1346279, upper bound: 88.1480922
time: 100.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.1206876, upper bound: 88.1759553
time: 83.92 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -92.8157196, 36.9061699, -92.8177109, 36.9092140, -129.7249298, 129.7238770
1: -59.5394745, 29.8969383, -59.5416145, 29.8988476, -87.8223419, 87.8353348
2: -49.5675964, 30.8797989, -49.5723381, 30.8825264, -80.4501190, 80.4521332
3: -62.4342918, 31.2808552, -62.4367599, 31.2842789, -93.7185669, 93.7176132
4: -55.0063171, 43.6798019, -55.0064316, 43.6816063, -98.6879272, 98.6862335
5: -67.3698883, 36.9511070, -67.3732071, 36.9541855, -104.3240738, 104.3243103
6: -69.0900421, 41.6060905, -69.1151886, 41.6054573, -110.6954956, 110.7212830
7: -86.7650375, 27.2567825, -86.7789917, 27.2606125, -112.9646759, 112.9798737
8: -64.9198151, 51.9131203, -64.9205017, 51.9176025, -116.8374176, 116.8336182
9: -39.1902618, 35.9052467, -39.1908493, 35.9090004, -75.0992584, 75.0960999
10: -77.3368759, 47.8117943, -77.3350830, 47.8150826, -125.1519623, 125.1468811
11: -93.1551819, 14.3183746, -93.1912079, 14.3224621, -103.8096542, 103.8168335
12: -58.9848595, 52.0721893, -58.9890442, 52.0720825, -111.0569458, 111.0612335
13: -64.1747284, 66.8663788, -64.1760864, 66.8652802, -131.0400085, 131.0424652
14: -142.6570129, 21.1010189, -142.6532593, 21.1083279, -163.7653351, 163.7542725
15: -56.0552330, 46.7620049, -56.0559196, 46.7690392, -102.8242722, 102.8179245
16: -85.6327820, 29.6842537, -85.6472931, 29.6867256, -113.1436920, 113.1296844
17: -157.8496094, 22.8913307, -157.8486328, 22.8967819, -165.3611298, 165.3744812
18: -75.2387238, 48.7832298, -75.2419434, 48.7882195, -124.0269470, 124.0251770
19: -60.4540176, 14.7683525, -60.4575691, 14.7693405, -75.2233582, 75.2259216
20: -51.7832489, 18.7846947, -51.7852554, 18.7875557, -70.5708008, 70.5699463
21: -78.6204758, 18.6870384, -78.6285553, 18.6892242, -96.1121216, 96.1006012
22: -82.8491058, 24.8937950, -82.8495026, 24.9119186, -107.7610245, 107.7433014
23: -52.0328598, 24.6006794, -52.0346947, 24.6015663, -76.6344299, 76.6353760
24: -50.2803535, 33.0998268, -50.2794762, 33.1036148, -83.3839722, 83.3793030
25: -46.4837189, 35.8315887, -46.4850349, 35.8496208, -82.3333435, 82.3166199
26: -77.1387024, 47.7795410, -77.1418152, 47.7833557, -124.9220581, 124.9213562
27: -73.1911316, 30.4727230, -73.1922302, 30.4739075, -103.6650391, 103.6649551
28: -58.1267281, 31.2646027, -58.1272163, 31.2664719, -89.3932037, 89.3918152
29: -91.7551575, 15.9508095, -91.7542725, 15.9541512, -106.2199249, 106.1896820
30: -69.9712448, 37.2893333, -69.9716339, 37.2934837, -107.2647247, 107.2609711
31: -66.2721405, 18.7402573, -66.2731552, 18.7422504, -85.0143890, 85.0134125
32: -67.5047455, 42.2801437, -67.5204010, 42.2765770, -109.7813263, 109.8005447
33: -58.9034004, 82.5061188, -58.9071274, 82.5110016, -139.2142334, 139.2077789
34: -60.6006699, 69.1546249, -60.6047020, 69.1523895, -127.6706085, 127.6626129
35: -50.8164635, 76.8054504, -50.8193054, 76.8066711, -126.8269653, 126.8266678
36: -60.7405930, 67.2193146, -60.7449875, 67.2113037, -127.5973663, 127.6012878
37: -56.5739441, 61.3053055, -56.5832825, 61.3073883, -116.4695663, 116.4728851
38: -75.5187225, 82.5965271, -75.5260468, 82.5890503, -157.9360962, 157.9594727
39: -64.1804352, 88.0075836, -64.1837769, 88.0128632, -149.5012817, 149.5062866
40: -59.4817009, 57.9469147, -59.4884796, 57.9449501, -117.4266510, 117.4353943
41: -50.9660530, 44.4428406, -50.9909134, 44.4399529, -95.4060059, 95.4337540
42: -56.2134171, 40.3490791, -56.2305374, 40.3494492, -96.5628662, 96.5796204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=438, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 322

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1346279, upper bound: 88.1455580
time: 111.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1206876, upper bound: 88.1742955
time: 98.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 212.87 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 212.87
Output dim: 37, lower bound: -88.1346279, upper bound: 88.1480922
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 212.87
Output dim: 37, lower bound: -88.1206876, upper bound: 88.1759553
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 212.87
Output dim: 37, lower bound: -88.1346279, upper bound: 88.1455580
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 212.87
Output dim: 37, lower bound: -88.1206876, upper bound: 88.1742955

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -92.8859863, 36.9815826, -92.5515594, 36.7714462, -129.6574402, 129.5331421
1: -59.5427208, 30.0479336, -59.3725243, 29.7715225, -87.7028046, 87.8430786
2: -49.5566711, 30.9315948, -49.4406929, 30.7801456, -80.3368149, 80.3722839
3: -62.4672012, 31.3524704, -62.3794670, 31.1737175, -93.6409149, 93.7319336
4: -54.9748764, 43.8199501, -54.8256721, 43.5382690, -98.5131454, 98.6456223
5: -67.3765564, 37.0070114, -67.2707672, 36.8357773, -104.2123337, 104.2777786
6: -69.2826385, 41.5861130, -68.9894714, 41.3995972, -110.6822357, 110.5755844
7: -86.7249908, 27.3733692, -86.6142807, 27.1300201, -112.7982635, 112.9445267
8: -64.8631592, 52.0533142, -64.6973724, 51.7625809, -116.6257401, 116.7506866
9: -39.2738686, 36.0293198, -39.0907211, 35.7884445, -75.0623169, 75.1200409
10: -77.2838287, 48.0389404, -77.0820160, 47.6175308, -124.9013596, 125.1209564
11: -93.1736450, 14.5060425, -93.0552673, 14.2115841, -103.7112122, 103.9054413
12: -59.0283127, 52.1040802, -58.8795395, 52.0013199, -111.0296326, 110.9836197
13: -64.5016708, 66.8319778, -63.9868050, 66.6165848, -131.1182556, 130.8187866
14: -142.6112823, 21.3878021, -142.2498474, 20.9299412, -163.5412292, 163.6376495
15: -56.0282784, 46.9488525, -55.8680420, 46.6115799, -102.6398621, 102.8168945
16: -85.6671143, 29.8637352, -85.4421844, 29.5240040, -113.0104218, 113.1369553
17: -157.7669373, 23.2300377, -157.4633789, 22.6664200, -165.0331573, 165.3943787
18: -75.2255402, 49.0180244, -75.0656128, 48.6643600, -123.8899002, 124.0836334
19: -60.4855766, 14.8417282, -60.3520889, 14.7038965, -75.1894760, 75.1938171
20: -51.8187561, 18.9131737, -51.6891975, 18.7176094, -70.5363617, 70.6023712
21: -78.6415176, 18.8400784, -78.4901657, 18.5906410, -96.0424652, 96.1288147
22: -82.8325958, 25.0439873, -82.6565628, 24.8169556, -107.6495514, 107.7005463
23: -52.0783997, 24.6980629, -51.9388351, 24.5402889, -76.6186905, 76.6369019
24: -50.4092789, 33.2340546, -50.1276016, 33.0479088, -83.4571838, 83.3616562
25: -46.6084213, 35.9256210, -46.3412552, 35.7933350, -82.4017563, 82.2668762
26: -77.1683273, 48.0490150, -76.9720459, 47.6542015, -124.8225250, 125.0210571
27: -73.2138672, 30.6540756, -73.0709839, 30.3875389, -103.6014099, 103.7250595
28: -58.1561584, 31.3469734, -58.0456352, 31.2063637, -89.3625183, 89.3926086
29: -91.7441177, 16.1274166, -91.5350113, 15.8386497, -106.0928040, 106.1865234
30: -70.0218353, 37.4631805, -69.8472824, 37.2084846, -107.2303162, 107.3104630
31: -66.3487549, 18.8338432, -66.1659393, 18.6640759, -85.0128326, 84.9997864
32: -67.6869965, 42.2938995, -67.3648071, 42.0559998, -109.7429962, 109.6587067
33: -59.2541389, 82.4213028, -58.6527710, 82.2413483, -139.3109741, 138.8625793
34: -60.7793617, 69.1202087, -60.4399414, 68.9526825, -127.6713333, 127.4552078
35: -51.1140976, 76.7259979, -50.6092300, 76.5556641, -126.8950272, 126.5302353
36: -61.0180359, 67.1063309, -60.5343781, 66.8869171, -127.5661392, 127.2728424
37: -56.8851547, 61.2685852, -56.3330345, 61.1826363, -116.6692810, 116.1813736
38: -75.8800125, 82.4981689, -75.2525940, 82.2265167, -157.9609680, 157.5802155
39: -64.6056747, 87.8781662, -63.8807716, 87.6630096, -149.6094971, 149.0639648
40: -59.6586494, 57.9121590, -59.3238945, 57.7425232, -117.4011688, 117.2360535
41: -51.1688843, 44.4272194, -50.8585281, 44.2737923, -95.4426727, 95.2857513
42: -56.3334694, 40.3569412, -56.1382103, 40.2103958, -96.5438690, 96.4951477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=438, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1141
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1207
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1125
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1195
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1142
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1126
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1180
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1179
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1164
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1208
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1163
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1127
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1148
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1143
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1156
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 479
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1160
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1326
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1147
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1530
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1144
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 322

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 729

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.0487631, upper bound: 88.1759551
time: 239.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.0487631, upper bound: 88.1759554
time: 118.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 360.04 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 360.04
Output dim: 37, lower bound: -88.0487631, upper bound: 88.1759551
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 360.04
Output dim: 37, lower bound: -88.0487631, upper bound: 88.1759554

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -92.8859863, 36.9815826, -92.3032227, 36.5878601, -129.4738464, 129.2848053
1: -59.5427208, 30.0479336, -59.1995125, 29.5949135, -87.5636749, 87.6652069
2: -49.5566711, 30.9315948, -49.2658386, 30.6435661, -80.2002411, 80.1974335
3: -62.4672012, 31.3524704, -62.2028046, 31.0177536, -93.4849548, 93.5552750
4: -54.9748764, 43.8199501, -54.7038383, 43.4251328, -98.4000092, 98.5237885
5: -67.3765564, 37.0070114, -67.1015930, 36.7083740, -104.0849304, 104.1086044
6: -69.2826385, 41.5861130, -68.8855057, 41.3432808, -110.6259155, 110.4716187
7: -86.7249908, 27.3733692, -86.3853531, 26.9237881, -112.6276093, 112.7075272
8: -64.8631592, 52.0533142, -64.4385986, 51.5396500, -116.4028091, 116.4919128
9: -39.2738686, 36.0293198, -39.0226288, 35.7215233, -74.9953918, 75.0519485
10: -77.2838287, 48.0389404, -76.9642487, 47.5208778, -124.8047028, 125.0031891
11: -93.1736450, 14.5060425, -92.8407211, 14.1190758, -103.6343079, 103.6930084
12: -59.0283127, 52.1040802, -58.7539673, 51.8541412, -110.8824539, 110.8580475
13: -64.5016708, 66.8319778, -63.9264069, 66.5106812, -131.0123596, 130.7583923
14: -142.6112823, 21.3878021, -141.9392700, 20.7620316, -163.3733215, 163.3270721
15: -56.0282784, 46.9488525, -55.7401276, 46.5302925, -102.5585709, 102.6889801
16: -85.6671143, 29.8637352, -85.3152466, 29.4217987, -112.9165573, 113.0165176
17: -157.7669373, 23.2300377, -157.2798767, 22.5321960, -164.9956055, 165.2036591
18: -75.2255402, 49.0180244, -74.8973312, 48.4721069, -123.6976471, 123.9153595
19: -60.4855766, 14.8417282, -60.2828026, 14.6554375, -75.1410141, 75.1245270
20: -51.8187561, 18.9131737, -51.5953140, 18.6548290, -70.4735870, 70.5084839
21: -78.6415176, 18.8400784, -78.3879547, 18.5332146, -95.9859772, 96.0307617
22: -82.8325958, 25.0439873, -82.5309677, 24.6845913, -107.5171890, 107.5749512
23: -52.0783997, 24.6980629, -51.8699188, 24.4992180, -76.5776215, 76.5679779
24: -50.4092789, 33.2340546, -50.0159111, 32.9583130, -83.3675919, 83.2499695
25: -46.6084213, 35.9256210, -46.1984177, 35.6057816, -82.2142029, 82.1240387
26: -77.1683273, 48.0490150, -76.7468262, 47.5207748, -124.6891022, 124.7958374
27: -73.2138672, 30.6540756, -72.9866409, 30.3463020, -103.5601654, 103.6407166
28: -58.1561584, 31.3469734, -57.9597626, 31.1532555, -89.3094177, 89.3067322
29: -91.7441177, 16.1274166, -91.4150848, 15.7694683, -106.0361557, 106.0572052
30: -70.0218353, 37.4631805, -69.7361221, 37.1090622, -107.1308975, 107.1993027
31: -66.3487549, 18.8338432, -66.0534363, 18.5828419, -84.9315948, 84.8872833
32: -67.6869965, 42.2938995, -67.2860641, 41.9958191, -109.6828156, 109.5799637
33: -59.2541389, 82.4213028, -58.4901085, 82.0154877, -139.0816956, 138.7201843
34: -60.7793617, 69.1202087, -60.3012886, 68.7911377, -127.4961700, 127.3548889
35: -51.1140976, 76.7259979, -50.5245247, 76.4562073, -126.8164368, 126.4786530
36: -61.0180359, 67.1063309, -60.3865318, 66.8018036, -127.4715805, 127.1466522
37: -56.8851547, 61.2685852, -56.1089554, 61.0113716, -116.4931488, 115.9709930
38: -75.8800125, 82.4981689, -75.1272659, 82.1598206, -157.9004517, 157.4743958
39: -64.6056747, 87.8781662, -63.7803345, 87.5325775, -149.4927673, 148.9922028
40: -59.6586494, 57.9121590, -59.1925964, 57.6575546, -117.3162079, 117.1047516
41: -51.1688843, 44.4272194, -50.7487564, 44.2343559, -95.4032440, 95.1759796
42: -56.3334694, 40.3569412, -56.0593185, 40.1475487, -96.4810181, 96.4162598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=490, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 602

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.9864624, upper bound: 88.1612715
time: 91.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.0418403, upper bound: 88.1734694
time: 211.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -92.8859863, 36.9815826, -92.5436401, 36.7670059, -129.6529846, 129.5252228
1: -59.5427208, 30.0479336, -59.3657951, 29.7685127, -87.6992416, 87.7938995
2: -49.5566711, 30.9315948, -49.4340401, 30.7761650, -80.3328400, 80.3656311
3: -62.4672012, 31.3524704, -62.3714561, 31.1684475, -93.6356506, 93.7239227
4: -54.9748764, 43.8199501, -54.8202133, 43.5345879, -98.5094604, 98.6401672
5: -67.3765564, 37.0070114, -67.2635345, 36.8309364, -104.2074890, 104.2705460
6: -69.2826385, 41.5861130, -68.9628525, 41.3938332, -110.6764679, 110.5489655
7: -86.7249908, 27.3733692, -86.5968933, 27.1247425, -112.7923889, 112.8923340
8: -64.8631592, 52.0533142, -64.6900024, 51.7561417, -116.6193008, 116.7433167
9: -39.2738686, 36.0293198, -39.0876694, 35.7832642, -75.0571289, 75.1169891
10: -77.2838287, 48.0389404, -77.0765381, 47.6118851, -124.8957138, 125.1154785
11: -93.1736450, 14.5060425, -93.0150909, 14.2063274, -103.7065277, 103.8519363
12: -59.0283127, 52.1040802, -58.8733177, 51.9959564, -111.0242691, 110.9774017
13: -64.5016708, 66.8319778, -63.9832115, 66.6101227, -131.1117859, 130.8151855
14: -142.6112823, 21.3878021, -142.2409668, 20.9216366, -163.5329132, 163.6287689
15: -56.0282784, 46.9488525, -55.8619652, 46.6030121, -102.6312866, 102.8108215
16: -85.6671143, 29.8637352, -85.4216309, 29.5197067, -113.0061493, 113.1057053
17: -157.7669373, 23.2300377, -157.4529877, 22.6590500, -165.0251007, 165.2816772
18: -75.2255402, 49.0180244, -75.0571442, 48.6578827, -123.8834229, 124.0751648
19: -60.4855766, 14.8417282, -60.3454819, 14.7020378, -75.1876144, 75.1872101
20: -51.8187561, 18.9131737, -51.6851692, 18.7136593, -70.5324173, 70.5983429
21: -78.6415176, 18.8400784, -78.4776154, 18.5875969, -96.0394897, 96.1155548
22: -82.8325958, 25.0439873, -82.6499176, 24.7978840, -107.6304779, 107.6939087
23: -52.0783997, 24.6980629, -51.9342957, 24.5383358, -76.6167374, 76.6323547
24: -50.4092789, 33.2340546, -50.1234055, 33.0434113, -83.4526901, 83.3574600
25: -46.6084213, 35.9256210, -46.3373070, 35.7743378, -82.3827591, 82.2629242
26: -77.1683273, 48.0490150, -76.9645538, 47.6488533, -124.8171844, 125.0135651
27: -73.2138672, 30.6540756, -73.0646744, 30.3852673, -103.5991364, 103.7187500
28: -58.1561584, 31.3469734, -58.0424728, 31.2031765, -89.3593369, 89.3894501
29: -91.7441177, 16.1274166, -91.5283203, 15.8343887, -106.0880432, 106.1580887
30: -70.0218353, 37.4631805, -69.8426208, 37.2030563, -107.2248917, 107.3058014
31: -66.3487549, 18.8338432, -66.1626892, 18.6610661, -85.0098190, 84.9965363
32: -67.6869965, 42.2938995, -67.3465271, 42.0528488, -109.7398453, 109.6404266
33: -59.2541389, 82.4213028, -58.6463509, 82.2279663, -139.2734985, 138.8561096
34: -60.7793617, 69.1202087, -60.4336700, 68.9486542, -127.6221008, 127.4482956
35: -51.1140976, 76.7259979, -50.6040306, 76.5469513, -126.8593750, 126.5246124
36: -61.0180359, 67.1063309, -60.5279541, 66.8847885, -127.5375214, 127.2659912
37: -56.8851547, 61.2685852, -56.3200760, 61.1763687, -116.6471558, 116.1681824
38: -75.8800125, 82.4981689, -75.2428894, 82.2222137, -157.9321899, 157.5701904
39: -64.6056747, 87.8781662, -63.8737488, 87.6462402, -149.5657806, 149.0570679
40: -59.6586494, 57.9121590, -59.3141708, 57.7382965, -117.3969421, 117.2263336
41: -51.1688843, 44.4272194, -50.8320312, 44.2714500, -95.4403381, 95.2592468
42: -56.3334694, 40.3569412, -56.1199074, 40.2054291, -96.5388947, 96.4768524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=437, inp2_unstable=437, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1141
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1207
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1125
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1195
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1142
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1126
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1180
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1179
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1164
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1208
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1163
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1127
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1148
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1143
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1156
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 479
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1160
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1326
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1147
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1530
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1144
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 602

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 574

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.9864624, upper bound: 88.1612719
time: 175.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.0418403, upper bound: 88.1734697
time: 89.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 267.91 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 267.91
Output dim: 37, lower bound: -87.9864624, upper bound: 88.1612715
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 267.91
Output dim: 37, lower bound: -88.0418403, upper bound: 88.1734694
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 267.91
Output dim: 37, lower bound: -87.9864624, upper bound: 88.1612719
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 267.91
Output dim: 37, lower bound: -88.0418403, upper bound: 88.1734697
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=116.50578308105469
rel_dist={37: [-88.24876147168659, 88.24876147750362]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 10288.02 seconds
