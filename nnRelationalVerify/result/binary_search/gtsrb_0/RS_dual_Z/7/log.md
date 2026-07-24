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
execution time: IAR + LP analysis = 2.93 + 53.94 = 56.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 37, lower bound: -96.1728623, upper bound: 96.1728623


# Binary Search by BASE starts (time budget: 17943.13 seconds, max iter: 100)

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
Binary search time: 498.89 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 17444.24 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3781659, upper bound: 92.2961246
time: 112.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2961246, upper bound: 92.3781659
time: 106.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 218.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 218.69
Output dim: 37, lower bound: -92.3781659, upper bound: 92.2961246
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 218.69
Output dim: 37, lower bound: -92.2961246, upper bound: 92.3781659

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9698639, 106.9701996
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3501282, 115.3501282
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.3403931, 170.3409576
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3688965, 140.3687592
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.5192871, 129.5190887
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1996689, 117.1995926
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5805664, 151.5803223
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3564630, upper bound: 92.1918769
time: 118.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2742732, upper bound: 92.2745968
time: 112.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9701996, 106.9698486
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3501282, 115.3501282
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.3409424, 170.3404083
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3687744, 140.3688812
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.5191040, 129.5192871
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1995926, 117.1996689
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5803528, 151.5805664
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2745968, upper bound: 92.2742732
time: 114.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1918769, upper bound: 92.3564630
time: 113.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 230.64 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 230.64
Output dim: 37, lower bound: -92.3564630, upper bound: 92.1918769
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 230.64
Output dim: 37, lower bound: -92.2742732, upper bound: 92.2745968
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 230.64
Output dim: 37, lower bound: -92.2745968, upper bound: 92.2742732
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 230.64
Output dim: 37, lower bound: -92.1918769, upper bound: 92.3564630

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9120636, 106.9383774
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3206024, 115.3391800
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2288818, 170.2702026
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3526917, 140.3432007
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4931030, 129.4777985
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1895599, 117.1836700
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5512085, 151.5340576
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3206668, upper bound: 92.1118266
time: 120.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2771736, upper bound: 92.1553926
time: 99.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9380341, 106.9124222
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3389435, 115.3208542
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2696533, 170.2294464
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3433533, 140.3525543
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4780121, 129.4929047
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1837616, 117.1894913
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5342865, 151.5509796
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2375593, upper bound: 92.1952240
time: 102.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1118266, upper bound: 92.2388234
time: 96.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9124298, 106.9380264
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3208466, 115.3389359
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2294312, 170.2696533
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3525696, 140.3433228
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4929047, 129.4780121
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1894989, 117.1837463
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5509644, 151.5342865
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2388234, upper bound: 92.1943046
time: 113.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1952240, upper bound: 92.2375593
time: 236.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9383698, 106.9120712
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3391876, 115.3206100
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2702026, 170.2288971
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3432312, 140.3526764
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4777985, 129.4931030
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1836700, 117.1895676
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5340576, 151.5512085
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1553926, upper bound: 92.2771736
time: 166.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1118266, upper bound: 92.3206668
time: 116.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 285.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.3206668, upper bound: 92.1118266
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.2771736, upper bound: 92.1553926
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.2375593, upper bound: 92.1952240
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.1118266, upper bound: 92.2388234
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.2388234, upper bound: 92.1943046
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.1952240, upper bound: 92.2375593
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.1553926, upper bound: 92.2771736
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 285.03
Output dim: 37, lower bound: -92.1118266, upper bound: 92.3206668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.8814545, 106.9303894
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.2987518, 115.3333130
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.1802673, 170.2571411
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3497009, 140.3320618
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4882660, 129.4597931
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1876907, 117.1767273
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5457764, 151.5139008
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3187946, upper bound: 92.0409627
time: 95.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2477260, upper bound: 92.1098731
time: 96.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9040833, 106.9077606
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3147430, 115.3173218
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2158508, 170.2216187
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3415527, 140.3402100
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4750824, 129.4729614
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1826248, 117.1818085
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5310364, 151.5286407
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2753006, upper bound: 92.0846200
time: 106.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2041515, upper bound: 92.1534400
time: 95.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9074097, 106.9044342
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3170776, 115.3149872
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2210388, 170.2164001
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3403625, 140.3414154
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4731598, 129.4748993
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1818924, 117.1825485
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5288696, 151.5308075
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2356828, upper bound: 92.1244801
time: 89.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1643806, upper bound: 92.1932732
time: 85.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9300537, 106.8818054
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3330688, 115.2989960
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2566223, 170.1808472
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3322144, 140.3495636
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4599915, 129.4880676
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1767960, 117.1876297
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5141296, 151.5455475
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1924198, upper bound: 92.1680963
time: 122.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1208188, upper bound: 92.2368784
time: 154.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.8818054, 106.9300537
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.2989960, 115.3330688
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.1808167, 170.2565918
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3495789, 140.3321838
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4880524, 129.4599915
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1876297, 117.1768036
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5455627, 151.5141144
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2368784, upper bound: 92.1208188
time: 150.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1680963, upper bound: 92.1924198
time: 133.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9044342, 106.9074097
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3149872, 115.3170776
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2164001, 170.2210693
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3414307, 140.3403320
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4748993, 129.4731598
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1825333, 117.1818848
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5307922, 151.5288696
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1932732, upper bound: 92.1643806
time: 127.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1244801, upper bound: 92.2356828
time: 133.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9077606, 106.9040833
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3173218, 115.3147430
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2215881, 170.2158508
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3402405, 140.3415375
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4729614, 129.4750977
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1818008, 117.1826324
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5286560, 151.5310364
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1534400, upper bound: 92.2041515
time: 152.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.0846200, upper bound: 92.2753006
time: 104.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.9303894, 106.8814545
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.3333130, 115.2987518
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.2571716, 170.1802979
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3320923, 140.3496857
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4597931, 129.4882660
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1767349, 117.1877060
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5138855, 151.5457916
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.0409627, upper bound: 92.2477260
time: 93.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.0409627, upper bound: 92.3187946
time: 120.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 215.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.3187946, upper bound: 92.0409627
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.2477260, upper bound: 92.1098731
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.2753006, upper bound: 92.0846200
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.2041515, upper bound: 92.1534400
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.2356828, upper bound: 92.1244801
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1643806, upper bound: 92.1932732
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1924198, upper bound: 92.1680963
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1208188, upper bound: 92.2368784
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.2368784, upper bound: 92.1208188
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1680963, upper bound: 92.1924198
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1932732, upper bound: 92.1643806
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1244801, upper bound: 92.2356828
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.1534400, upper bound: 92.2041515
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.0846200, upper bound: 92.2753006
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.0409627, upper bound: 92.2477260
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 215.88
Output dim: 37, lower bound: -92.0409627, upper bound: 92.3187946

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.8807144, 106.9295807
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.2993546, 115.3338547
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.1823120, 170.2590637
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3501434, 140.3325500
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4889984, 129.4605560
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1879654, 117.1770172
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5466003, 151.5147858
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.3152695, upper bound: 91.9168499
time: 101.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1955196, upper bound: 92.0374256
time: 88.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -89.4498901, 89.4498901
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -114.0523911, 114.0523911
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -106.8806381, 106.9296570
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -115.2992935, 115.3339005
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -170.1821899, 170.2591553
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -97.3284912, 97.3284912
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.7217255, 107.7217255
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -140.3501740, 140.3325195
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -129.4890289, 129.4605103
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.6423645, 127.6423645
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -117.1879959, 117.1770020
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -151.5466614, 151.5147247
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.2441978, upper bound: 91.9859679
time: 109.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -92.1239534, upper bound: 92.1063427
time: 122.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 234.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 234.21
Output dim: 37, lower bound: -92.3152695, upper bound: 91.9168499
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 234.21
Output dim: 37, lower bound: -92.1955196, upper bound: 92.0374256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 234.21
Output dim: 37, lower bound: -92.2441978, upper bound: 91.9859679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 234.21
Output dim: 37, lower bound: -92.1239534, upper bound: 92.1063427
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.2753006, upper bound: 92.0846200
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.2041515, upper bound: 92.1534400
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.2356828, upper bound: 92.1244801
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1643806, upper bound: 92.1932732
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1924198, upper bound: 92.1680963
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1208188, upper bound: 92.2368784
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.2368784, upper bound: 92.1208188
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1680963, upper bound: 92.1924198
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1932732, upper bound: 92.1643806
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1244801, upper bound: 92.2356828
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.1534400, upper bound: 92.2041515
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.0846200, upper bound: 92.2753006
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.0409627, upper bound: 92.2477260
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 234.21
Output dim: 37, lower bound: -92.0409627, upper bound: 92.3187946
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=117.19937133789062
rel_dist={37: [-92.38023227333375, 92.38023227136748]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6778294, upper bound: 89.6158866
time: 98.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6158866, upper bound: 89.6778294
time: 79.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 178.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 178.75
Output dim: 37, lower bound: -89.6778294, upper bound: 89.6158866
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 178.75
Output dim: 37, lower bound: -89.6158866, upper bound: 89.6778294

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4511414, 88.4512939
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5673294, 113.5674744
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.9082489, 104.9085236
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.9149170, 113.9151001
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.1035156, 167.1039124
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6279297, 96.6280518
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1961517, 107.1964111
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6268005, 139.6266937
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.3198242, 128.3196716
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3597794, 127.3596573
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7372818, 116.7372208
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.2373047, 150.2371216
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6555545, upper bound: 89.5314253
time: 107.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5933950, upper bound: 89.5936937
time: 89.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4512939, 88.4511414
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5674820, 113.5673370
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.9085236, 104.9082489
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.9151001, 113.9149170
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.1039429, 167.1034851
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6280518, 96.6279221
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1964111, 107.1961594
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6267090, 139.6267853
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.3196716, 128.3198242
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3596573, 127.3597794
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9733887, 127.9733887
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7372208, 116.7372818
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.2371521, 150.2373047
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5936938, upper bound: 89.5933950
time: 103.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5314253, upper bound: 89.6555545
time: 105.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 211.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 211.34
Output dim: 37, lower bound: -89.6555545, upper bound: 89.5314253
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 211.34
Output dim: 37, lower bound: -89.5933950, upper bound: 89.5936937
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 211.34
Output dim: 37, lower bound: -89.5936938, upper bound: 89.5933950
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 211.34
Output dim: 37, lower bound: -89.5314253, upper bound: 89.6555545

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4113770, 88.4227829
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5287170, 113.5394592
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8504486, 104.8709183
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8647308, 113.8791885
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9919739, 167.0241089
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6002045, 96.6100464
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1319427, 107.1505966
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6085205, 139.6011353
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2902832, 128.2783813
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3358459, 127.3262177
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9672852, 127.9603882
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7258759, 116.7212982
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.2041779, 150.1908417
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6196018, upper bound: 89.4614164
time: 84.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5858385, upper bound: 89.4944890
time: 89.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4226379, 88.4115219
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5393066, 113.5288620
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8706512, 104.8507309
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8789825, 113.8649368
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.0236816, 166.9924011
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6099091, 96.6003265
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1503448, 107.1322021
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6012573, 139.6083984
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2785492, 128.2901306
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3263397, 127.3357239
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9604797, 127.9671783
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7213593, 116.7258224
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1910248, 150.2040100
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5563565, upper bound: 89.5240740
time: 101.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4614164, upper bound: 89.5580029
time: 107.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4115295, 88.4226303
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5288696, 113.5393219
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8507233, 104.8706436
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8649445, 113.8790054
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9924011, 167.0236816
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6003265, 96.6099091
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1322021, 107.1503525
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6084290, 139.6012268
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2901306, 128.2785492
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3357239, 127.3263474
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9671936, 127.9604797
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7258148, 116.7213593
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.2040100, 150.1910248
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5580029, upper bound: 89.5233113
time: 119.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5240740, upper bound: 89.5563565
time: 102.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4227905, 88.4113693
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5394592, 113.5287247
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8709259, 104.8504562
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8791962, 113.8647385
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.0241089, 166.9919739
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6100464, 96.6002045
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1505890, 107.1319504
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6011353, 139.6084900
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2783813, 128.2902832
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3262177, 127.3358459
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9603882, 127.9672699
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7212982, 116.7258835
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1908417, 150.2041779
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4944890, upper bound: 89.5858385
time: 91.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4614164, upper bound: 89.6196018
time: 111.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 205.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.6196018, upper bound: 89.4614164
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.5858385, upper bound: 89.4944890
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.5563565, upper bound: 89.5240740
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.4614164, upper bound: 89.5580029
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.5580029, upper bound: 89.5233113
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.5240740, upper bound: 89.5563565
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.4944890, upper bound: 89.5858385
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.46
Output dim: 37, lower bound: -89.4614164, upper bound: 89.6196018

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.3969116, 88.4181366
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5189362, 113.5389175
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8198395, 104.8579102
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8428802, 113.8697662
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9433899, 167.0031433
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5854645, 96.6037979
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1037445, 107.1384430
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6037292, 139.5899963
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2825165, 128.2603760
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3295746, 127.3116379
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9627991, 127.9499817
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7228775, 116.7143555
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1954956, 150.1706696
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6181611, upper bound: 89.4064267
time: 92.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5657583, upper bound: 89.4599559
time: 98.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4067383, 88.4083252
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5281677, 113.5296707
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8374481, 104.8403015
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8553162, 113.8573303
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9710388, 166.9755249
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5939331, 96.5953140
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1197815, 107.1223907
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5973816, 139.5963440
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2722626, 128.2706146
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3212738, 127.3199234
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9568634, 127.9559021
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7189407, 116.7182999
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1840210, 150.1821442
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5843781, upper bound: 89.4395825
time: 107.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5318125, upper bound: 89.4930292
time: 70.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4081726, 88.4068756
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5295410, 113.5283203
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8400269, 104.8377228
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8571320, 113.8555145
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9750671, 166.9714661
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5951843, 96.5940781
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1221466, 107.1200409
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5964661, 139.5972595
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2707672, 128.2721252
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3200531, 127.3211441
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9560089, 127.9567719
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7183609, 116.7188797
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1823120, 150.1838379
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5548916, upper bound: 89.4692301
time: 104.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5022729, upper bound: 89.5226199
time: 107.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4179993, 88.3970490
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5387726, 113.5190735
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8576355, 104.8201141
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8695831, 113.8430786
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.0027161, 166.9438171
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6036530, 96.5856094
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1381836, 107.1040039
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5901184, 139.6036072
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2605286, 128.2823639
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3117828, 127.3294296
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9500885, 127.9626923
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7144241, 116.7228317
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1708679, 150.1952972
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5218389, upper bound: 89.5032552
time: 89.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4691849, upper bound: 89.5565526
time: 52.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.3970642, 88.4179993
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5190735, 113.5387726
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8201141, 104.8576355
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8430786, 113.8695831
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9438171, 167.0027161
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5856018, 96.6036606
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1040039, 107.1381989
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6036072, 139.5900879
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2823639, 128.2605286
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3294220, 127.3117676
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9627075, 127.9500732
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7228165, 116.7144165
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1953125, 150.1708527
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5565526, upper bound: 89.4691849
time: 98.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5032552, upper bound: 89.5218389
time: 92.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4068756, 88.4081726
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5283203, 113.5295334
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8377228, 104.8400269
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8555145, 113.8571472
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9714661, 166.9750977
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5940704, 96.5951920
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1200409, 107.1221466
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5972900, 139.5964355
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2721100, 128.2707672
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3211517, 127.3200531
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9567719, 127.9559937
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7188797, 116.7183609
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1838379, 150.1823273
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5226199, upper bound: 89.5022729
time: 91.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4692301, upper bound: 89.5548916
time: 85.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4083252, 88.4067230
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5296783, 113.5281754
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8403015, 104.8374481
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8573303, 113.8553162
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9754944, 166.9710388
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5953064, 96.5939407
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1223907, 107.1197968
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5963440, 139.5973511
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2706146, 128.2722778
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3199310, 127.3212662
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9559174, 127.9568634
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7182999, 116.7189407
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1821594, 150.1840057
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4930292, upper bound: 89.5318125
time: 145.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4395825, upper bound: 89.5843781
time: 94.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4181519, 88.3969116
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5389099, 113.5189362
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8579102, 104.8198395
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8697662, 113.8428802
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -167.0031433, 166.9433899
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.6037903, 96.5854721
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1384277, 107.1037445
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5900269, 139.6036987
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2603607, 128.2825165
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3116302, 127.3295593
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9499969, 127.9627838
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7143631, 116.7228928
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1706848, 150.1954803
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4599559, upper bound: 89.5657583
time: 100.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4064267, upper bound: 89.6181611
time: 109.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 212.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.6181611, upper bound: 89.4064267
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5657583, upper bound: 89.4599559
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5843781, upper bound: 89.4395825
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5318125, upper bound: 89.4930292
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5548916, upper bound: 89.4692301
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5022729, upper bound: 89.5226199
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5218389, upper bound: 89.5032552
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4691849, upper bound: 89.5565526
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5565526, upper bound: 89.4691849
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5032552, upper bound: 89.5218389
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.5226199, upper bound: 89.5022729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4692301, upper bound: 89.5548916
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4930292, upper bound: 89.5318125
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4395825, upper bound: 89.5843781
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4599559, upper bound: 89.5657583
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 212.68
Output dim: 37, lower bound: -89.4064267, upper bound: 89.6181611

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.3976517, 88.4188538
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5196381, 113.5395966
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8190842, 104.8571014
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8434830, 113.8703079
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9453735, 167.0050659
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5850143, 96.6033173
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1029510, 107.1375885
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6041565, 139.5904694
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2832489, 128.2611389
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3301468, 127.3122559
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9632263, 127.9504242
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7231522, 116.7146378
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1963196, 150.1715546
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.6155475, upper bound: 89.3109617
time: 308.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5235181, upper bound: 89.4037982
time: 87.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.3976212, 88.4188843
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5196075, 113.5396271
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8190231, 104.8571548
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8434219, 113.8703461
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9452820, 167.0051575
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5849991, 96.6033325
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1029053, 107.1376419
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.6041718, 139.5904541
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2832794, 128.2610931
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3301773, 127.3122330
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9632263, 127.9504089
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7231827, 116.7146301
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1963501, 150.1715240
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5631384, upper bound: 89.3648121
time: 129.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4709020, upper bound: 89.4573304
time: 121.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4074783, 88.4090271
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5288849, 113.5303497
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8366928, 104.8394928
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8559036, 113.8578720
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9730225, 166.9774170
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5934982, 96.5948334
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1190033, 107.1215439
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5978241, 139.5968018
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2729950, 128.2713776
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3218460, 127.3205490
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9572754, 127.9563446
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7192154, 116.7185898
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1848450, 150.1830292
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5817604, upper bound: 89.3442399
time: 91.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4895262, upper bound: 89.4369541
time: 100.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -88.4074326, 88.4090576
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.5288544, 113.5303802
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -104.8366318, 104.8395462
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.8558731, 113.8579102
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -166.9729309, 166.9775085
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.5934677, 96.5948639
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -107.1189423, 107.1215973
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.5978394, 139.5967865
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -128.2730255, 128.2713470
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -127.3218765, 127.3205185
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.9573059, 127.9563293
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.7192154, 116.7185745
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -158.1359558, 158.1359558
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -150.1848755, 150.1829834
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.5291909, upper bound: 89.3979902
time: 150.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -89.4368570, upper bound: 89.4904040
time: 89.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 242.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.6155475, upper bound: 89.3109617
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.5235181, upper bound: 89.4037982
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.5631384, upper bound: 89.3648121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.4709020, upper bound: 89.4573304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.5817604, upper bound: 89.3442399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.4895262, upper bound: 89.4369541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.5291909, upper bound: 89.3979902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.32
Output dim: 37, lower bound: -89.4368570, upper bound: 89.4904040
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5548916, upper bound: 89.4692301
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5022729, upper bound: 89.5226199
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5218389, upper bound: 89.5032552
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4691849, upper bound: 89.5565526
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5565526, upper bound: 89.4691849
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5032552, upper bound: 89.5218389
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.5226199, upper bound: 89.5022729
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4692301, upper bound: 89.5548916
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4930292, upper bound: 89.5318125
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4395825, upper bound: 89.5843781
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4599559, upper bound: 89.5657583
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 242.32
Output dim: 37, lower bound: -89.4064267, upper bound: 89.6181611
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=116.73698425292969
rel_dist={37: [-89.67935830190423, 89.67935830286939]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.2474023, upper bound: 88.1923496
time: 109.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.1923496, upper bound: 88.2474024
time: 102.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 212.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 212.42
Output dim: 37, lower bound: -88.2474023, upper bound: 88.1923496
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 212.42
Output dim: 37, lower bound: -88.1923496, upper bound: 88.2474024

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8760529, 87.8761902
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.0262833, 113.0264130
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8774414, 103.8776703
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1869812, 113.1871490
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.4850464, 165.4853973
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1318665, 96.1319809
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.2568207, 106.2570419
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2557678, 139.2556610
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.7200851, 127.7199631
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8744125, 126.8742981
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6374435, 127.6373520
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.5060806, 116.5060196
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9991913, 157.9990540
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.5656738, 149.5655212
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1721757, upper bound: 88.1171281
time: 110.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1721757, upper bound: 88.1697579
time: 123.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8761902, 87.8760529
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -113.0264053, 113.0262909
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8776703, 103.8774414
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1871490, 113.1869812
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.4854126, 165.4850311
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1319885, 96.1318665
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.2570343, 106.2568283
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2556763, 139.2557373
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.7199631, 127.7200928
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8743057, 126.8744125
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6373520, 127.6374283
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.5060196, 116.5060806
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9990540, 157.9992065
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.5655212, 149.5656738
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.1697579, upper bound: 88.1721757
time: 108.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.1171281, upper bound: 88.2247944
time: 105.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 216.15 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 216.15
Output dim: 37, lower bound: -88.1721757, upper bound: 88.1171281
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 216.15
Output dim: 37, lower bound: -88.1721757, upper bound: 88.1697579
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 216.15
Output dim: 37, lower bound: -88.1697579, upper bound: 88.1721757
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 216.15
Output dim: 37, lower bound: -88.1171281, upper bound: 88.2247944

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8460693, 87.8362885
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9968719, 112.9876709
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8372040, 103.8196487
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1491852, 113.1368179
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.4010620, 165.3735199
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1125946, 96.1041412
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.2085876, 106.1926193
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2301025, 139.2364197
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6786652, 127.6888733
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8408661, 126.8491211
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6134644, 127.6193390
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4900970, 116.4940414
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9546204, 157.9655914
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.5192413, 149.5306702
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.0803070, upper bound: 88.1601974
time: 97.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.0524175, upper bound: 88.1889349
time: 92.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 192.52 seconds
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 192.52
Output dim: 37, lower bound: -88.0803070, upper bound: 88.1601974
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 192.52
Output dim: 37, lower bound: -88.0524175, upper bound: 88.1889349

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8400269, 87.8218231
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9950256, 112.9778824
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8216705, 103.7890320
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1379852, 113.1149597
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3761292, 165.3249207
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1051178, 96.0894241
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1941528, 106.1644135
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2189941, 139.2307129
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6606598, 127.6796417
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8262787, 126.8416443
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6030426, 127.6140137
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4831619, 116.4904861
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9352264, 157.9556427
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4990845, 149.5203400
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.0511199, upper bound: 88.1414275
time: 89.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -88.0047294, upper bound: 88.1876212
time: 196.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 288.65 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 288.65
Output dim: 37, lower bound: -88.0511199, upper bound: 88.1414275
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 288.65
Output dim: 37, lower bound: -88.0047294, upper bound: 88.1876212

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8407364, 87.8225708
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9956818, 112.9785919
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8208542, 103.7882767
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1385269, 113.1155319
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3780518, 165.3269043
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1046524, 96.0889740
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1932983, 106.1636124
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2194366, 139.2311554
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6614227, 127.6803741
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8269119, 126.8422394
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6034851, 127.6144409
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4834366, 116.4907608
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9360657, 157.9564514
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4999390, 149.5211639
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -88.0024960, upper bound: 88.1068493
time: 79.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.9236786, upper bound: 88.1853916
time: 157.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 239.15 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 239.15
Output dim: 37, lower bound: -88.0024960, upper bound: 88.1068493
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 239.15
Output dim: 37, lower bound: -87.9236786, upper bound: 88.1853916

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8369446, 87.8180084
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9921265, 112.9743195
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8163147, 103.7823792
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1337433, 113.1097641
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3674164, 165.3141022
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1041107, 96.0877762
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1915131, 106.1605835
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2165070, 139.2286987
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6566925, 127.6764297
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8230743, 126.8390503
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6007233, 127.6121597
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4816208, 116.4892426
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9309692, 157.9521790
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4946289, 149.5167542
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.9223370, upper bound: 88.1268951
time: 590.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.8704252, upper bound: 88.1840491
time: 86.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 679.72 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 679.72
Output dim: 37, lower bound: -87.9223370, upper bound: 88.1268951
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 679.72
Output dim: 37, lower bound: -87.8704252, upper bound: 88.1840491

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8432007, 87.8235397
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9980011, 112.9794998
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8222961, 103.7870636
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1416397, 113.1167526
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3859863, 165.3306732
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1043930, 96.0874329
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1885147, 106.1564026
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2203522, 139.2330322
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6624527, 127.6829376
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8277206, 126.8443146
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.6040726, 127.6159210
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4842834, 116.4921799
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9371033, 157.9591827
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4981995, 149.5211792
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.8486222, upper bound: 88.1791629
time: 88.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.8628545, upper bound: 88.1233087
time: 110.90 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 201.34 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 201.34
Output dim: 37, lower bound: -87.8486222, upper bound: 88.1791629
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 201.34
Output dim: 37, lower bound: -87.8628545, upper bound: 88.1233087

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8390503, 87.8152237
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9942627, 112.9718323
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.8149948, 103.7722473
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1366348, 113.1064377
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3743591, 165.3072510
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.1009674, 96.0803986
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1817780, 106.1428223
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.2149811, 139.2303467
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6537552, 127.6786270
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.8207016, 126.8408203
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.5990677, 127.6134415
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4809418, 116.4905167
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.9277954, 157.9545441
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4884644, 149.5163116
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1581

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.8455643, upper bound: 88.1313417
time: 123.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.8140817, upper bound: 88.1776458
time: 85.32 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 211.14 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 211.14
Output dim: 37, lower bound: -87.8455643, upper bound: 88.1313417
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 211.14
Output dim: 37, lower bound: -87.8140817, upper bound: 88.1776458

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8160706, 87.7822266
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9723206, 112.9404755
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.7763672, 103.7156982
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1027298, 113.0598907
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3007050, 165.2054596
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.0826797, 96.0534821
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1400909, 106.0847931
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.1916351, 139.2134705
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6160278, 127.6513138
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.7902069, 126.8187485
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.5771713, 127.5975952
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4663849, 116.4799957
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.8872223, 157.9251862
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4462280, 149.4857788
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.8027440, upper bound: 88.1772369
time: 97.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.8136671, upper bound: 88.1666044
time: 82.52 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 182.67 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 182.67
Output dim: 37, lower bound: -87.8027440, upper bound: 88.1772369
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 182.67
Output dim: 37, lower bound: -87.8136671, upper bound: 88.1666044

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8188095, 87.7849274
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9744263, 112.9425507
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.7804031, 103.7196655
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1062317, 113.0633316
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3025360, 165.2071533
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.0836487, 96.0544281
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1412430, 106.0858841
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.1934052, 139.2152557
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6183014, 127.6536255
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.7924576, 126.8210602
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.5788269, 127.5992584
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4673309, 116.4809418
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.8894043, 157.9274292
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4494324, 149.4889832
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1761

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.7995133, upper bound: 88.1269810
time: 124.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 37, lower bound: -87.7743644, upper bound: 88.1761066
time: 102.19 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 228.94 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 228.94
Output dim: 37, lower bound: -87.7995133, upper bound: 88.1269810
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 228.94
Output dim: 37, lower bound: -87.7743644, upper bound: 88.1761066

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -92.8269806, 36.9121208, -92.8269806, 36.9121208, -129.7391052, 129.7391052
1: -59.5488739, 29.9010124, -59.5488739, 29.9010124, -87.8180237, 87.7840729
2: -49.5766220, 30.8851204, -49.5766220, 30.8851204, -80.4617462, 80.4617462
3: -62.4454842, 31.2880211, -62.4454842, 31.2880211, -93.7335052, 93.7335052
4: -55.0142899, 43.6848602, -55.0142899, 43.6848602, -98.6991501, 98.6991501
5: -67.3801270, 36.9576530, -67.3801270, 36.9576530, -104.3377838, 104.3377838
6: -69.1245422, 41.6147308, -69.1245422, 41.6147308, -110.7392731, 110.7392731
7: -86.7885284, 27.2638626, -86.7885284, 27.2638626, -112.9737015, 112.9417419
8: -64.9305267, 51.9217682, -64.9305267, 51.9217682, -116.8522949, 116.8522949
9: -39.1947594, 35.9121933, -39.1947594, 35.9121933, -75.1069489, 75.1069489
10: -77.3452911, 47.8195038, -77.3452911, 47.8195038, -125.1647949, 125.1647949
11: -93.2068634, 14.3253803, -93.2068634, 14.3253803, -103.7793198, 103.7184601
12: -58.9932175, 52.0801201, -58.9932175, 52.0801201, -111.0733337, 111.0733337
13: -64.1798248, 66.8761292, -64.1798248, 66.8761292, -131.0559540, 131.0559540
14: -142.6708984, 21.1119289, -142.6708984, 21.1119289, -163.7828217, 163.7828217
15: -56.0640907, 46.7733612, -56.0640907, 46.7733612, -102.8374481, 102.8374481
16: -85.6600189, 29.6901054, -85.6600189, 29.6901054, -113.1052017, 113.0622253
17: -157.8650360, 22.9012489, -157.8650360, 22.9012489, -165.3003082, 165.2047424
18: -75.2505798, 48.7920532, -75.2505798, 48.7920532, -124.0426331, 124.0426331
19: -60.4632301, 14.7708912, -60.4632301, 14.7708912, -75.2341232, 75.2341232
20: -51.7888412, 18.7899876, -51.7888412, 18.7899876, -70.5788269, 70.5788269
21: -78.6373444, 18.6911507, -78.6373444, 18.6911507, -96.0831909, 96.0539093
22: -82.8589020, 24.9182854, -82.8589020, 24.9182854, -107.7771912, 107.7771912
23: -52.0391998, 24.6034031, -52.0391998, 24.6034031, -76.6426010, 76.6426010
24: -50.2867355, 33.1057358, -50.2867355, 33.1057358, -83.3924713, 83.3924713
25: -46.4892807, 35.8571930, -46.4892807, 35.8571930, -82.3464737, 82.3464737
26: -77.1492996, 47.7866287, -77.1492996, 47.7866287, -124.9359283, 124.9359283
27: -73.2002563, 30.4758568, -73.2002563, 30.4758568, -103.6761169, 103.6761169
28: -58.1312790, 31.2689457, -58.1312790, 31.2689457, -89.4002228, 89.4002228
29: -91.7652435, 15.9564838, -91.7652435, 15.9564838, -106.1405792, 106.0851059
30: -69.9780807, 37.2965050, -69.9780807, 37.2965050, -107.2745819, 107.2745819
31: -66.2767334, 18.7443485, -66.2767334, 18.7443485, -85.0210800, 85.0210800
32: -67.5284882, 42.2854538, -67.5284882, 42.2854538, -109.8139420, 109.8139420
33: -58.9121552, 82.5250702, -58.9121552, 82.5250702, -139.1928253, 139.2147217
34: -60.6091537, 69.1612701, -60.6091537, 69.1612701, -127.6173935, 127.6527939
35: -50.8236122, 76.8187485, -50.8236122, 76.8187485, -126.7917175, 126.8203888
36: -60.7492561, 67.2241364, -60.7492561, 67.2241364, -127.5783081, 127.5987778
37: -56.5913467, 61.3142395, -56.5913467, 61.3142395, -116.4669952, 116.4806442
38: -75.5316391, 82.6043243, -75.5316391, 82.6043243, -157.8884277, 157.9265137
39: -64.1901398, 88.0310287, -64.1901398, 88.0310287, -149.4483643, 149.4880524
40: -59.4948502, 57.9534836, -59.4948502, 57.9534836, -117.4483337, 117.4483337
41: -51.0001411, 44.4468384, -51.0001411, 44.4468384, -95.4469757, 95.4469757
42: -56.2370300, 40.3563385, -56.2370300, 40.3563385, -96.5933685, 96.5933685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=439, inp2_unstable=439, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=491, inp2_unstable=491, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 479
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1141
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1208
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1207
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1147
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1160
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1529
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1527
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1195
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1148
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1142
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1143
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1179
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 338
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1292
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 1144
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 447
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1156
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1180
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1111
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1096
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1164
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 1509
type: RSZ, layer: 1, pos: 1163
type: RSZ, layer: 1, pos: 1127
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1126
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1693

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.7186538, upper bound: 88.1239673
time: 103.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 37, lower bound: -87.7225984, upper bound: 88.1202494
time: 89.53 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 195.48 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 195.48
Output dim: 37, lower bound: -87.7186538, upper bound: 88.1239673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 195.48
Output dim: 37, lower bound: -87.7225984, upper bound: 88.1202494
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=116.50578308105469
rel_dist={37: [-88.24876147168659, 88.24876147750362]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 11556.96 seconds
