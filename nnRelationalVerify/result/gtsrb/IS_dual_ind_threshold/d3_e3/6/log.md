## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 144.732426197


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972)
1: (-77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194)
2: (-73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790)
3: (-78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870)
4: (-86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896)
5: (-83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952)
6: (-101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076)
7: (-104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297)
8: (-94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204)
9: (-79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501)
10: (-120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379)
11: (-121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262)
12: (-109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437)
13: (-116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750)
14: (-181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808)
15: (-88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771)
16: (-125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754)
17: (-189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771)
18: (-115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102)
19: (-85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734)
20: (-78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482)
21: (-112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949)
22: (-121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192)
23: (-88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307)
24: (-112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724)
25: (-93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010)
26: (-124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929)
27: (-118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892)
28: (-86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454)
29: (-133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488)
30: (-110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042)
31: (-110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487)
32: (-110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070)
33: (-150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553)
34: (-125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694)
35: (-127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004)
36: (-124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196)
37: (-169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105)
38: (-148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124)
39: (-176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072)
40: (-147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060)
41: (-110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491)
42: (-80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.99 + 451.52 = 454.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -144.8773035, upper bound: 144.8773035

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1206
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 349
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 399
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 300
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 317
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 316
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 365
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 268
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 278
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 267
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 284
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 285
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 286
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 351
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1125

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.6532863, upper bound: 144.8634749
time: 257.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8688623, upper bound: 144.8688632
time: 188.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 445.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 445.71
Output dim: 12, lower bound: -144.6532863, upper bound: 144.8634749
IS_A2, status: Status.UNKNOWN, split count: 1, time: 445.71
Output dim: 12, lower bound: -144.8688623, upper bound: 144.8688632

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -148.3609467, 80.5298615, -148.4949036, 80.6621399, -229.0230865, 229.0247650
1: -77.3296051, 74.2723083, -77.3897934, 74.3903656, -151.7199707, 151.6621094
2: -73.4323425, 65.8842850, -73.5816498, 66.0222168, -139.4545288, 139.4659271
3: -78.6891022, 82.7835007, -78.8569946, 82.9479523, -161.6370544, 161.6405029
4: -86.2105331, 81.9066010, -86.4142914, 82.1149445, -168.3254700, 168.3208923
5: -83.1830444, 87.0066376, -83.2666855, 87.1681824, -170.3512115, 170.2733154
6: -101.6542816, 87.7330627, -101.8139343, 87.9354553, -189.5897217, 189.5469971
7: -104.4187012, 88.7366562, -104.6136551, 88.9258728, -193.3445740, 193.3503113
8: -94.5821686, 101.6687317, -94.8050842, 101.8709869, -196.4531555, 196.4738159
9: -79.6185760, 82.1346970, -79.7189560, 82.2540588, -161.8726349, 161.8536377
10: -120.1875000, 113.4302826, -120.3974152, 113.7162170, -233.9037170, 233.8276825
11: -121.0341415, 90.5477829, -121.2146530, 90.6536331, -211.6877747, 211.7624207
12: -108.6586761, 114.4509888, -109.1389313, 115.0117874, -223.6704712, 223.5899200
13: -115.8065948, 122.2277527, -116.0231400, 122.5076904, -238.3142853, 238.2508850
14: -180.7997589, 111.8337021, -181.2164001, 112.2467499, -293.0464783, 293.0501099
15: -88.6489105, 80.9712677, -88.8485718, 81.0472107, -169.6961212, 169.8198395
16: -124.7506866, 92.2411575, -124.9677048, 92.4219513, -217.1726227, 217.2088623
17: -188.1927490, 152.3658142, -188.6625366, 152.8692627, -341.0620117, 341.0283508
18: -115.3403168, 89.4271851, -115.5516663, 89.5377960, -204.8781128, 204.9788361
19: -85.2268600, 41.4963722, -85.3294525, 41.5582886, -126.7851486, 126.8258133
20: -78.2730331, 56.3815613, -78.4282913, 56.4877434, -134.7607727, 134.8098450
21: -111.7395630, 60.1880341, -111.8984833, 60.2757607, -172.0152893, 172.0864868
22: -121.0644150, 73.4459381, -121.3636780, 73.7456055, -194.8099976, 194.8096161
23: -88.4911957, 58.8904495, -88.6462708, 58.9617615, -147.4529572, 147.5367126
24: -112.3549576, 63.0954704, -112.5827713, 63.2314606, -175.5864105, 175.6782379
25: -93.5551071, 66.0703049, -93.7032547, 66.1593704, -159.7144775, 159.7735596
26: -123.2594147, 103.1044006, -123.6623383, 103.3920822, -226.6514740, 226.7667389
27: -118.5722809, 83.5296936, -118.7461624, 83.6253128, -202.1975861, 202.2758484
28: -86.0430603, 66.3331909, -86.1704483, 66.4088058, -152.4518738, 152.5036316
29: -132.9238281, 81.5706024, -133.2547913, 81.9799194, -214.9037476, 214.8253632
30: -110.7493896, 85.3305740, -110.9080200, 85.4972687, -196.2466583, 196.2385864
31: -110.0999374, 56.0138435, -110.2967987, 56.1378479, -166.2377930, 166.3106384
32: -109.6594543, 87.1654663, -109.8626099, 87.4252472, -197.0847015, 197.0280762
33: -149.7644348, 108.4904099, -150.1511993, 108.7940063, -258.5584412, 258.6416016
34: -125.3485184, 92.7162476, -125.6165237, 92.8707733, -218.2192688, 218.3327637
35: -127.5535278, 90.7922592, -127.8356934, 90.9578552, -218.5113831, 218.6279602
36: -123.6609039, 98.5469513, -123.8583374, 98.7570419, -222.4179382, 222.4052887
37: -169.3692322, 97.4917908, -169.6730957, 97.6183701, -266.9876099, 267.1648865
38: -147.8439484, 120.5947495, -148.0698700, 120.7971115, -268.6410522, 268.6646118
39: -176.0461273, 113.2717590, -176.3322144, 113.5074768, -289.5535889, 289.6039429
40: -146.9808655, 94.8457031, -147.2586060, 95.0393982, -242.0202637, 242.1042786
41: -109.8422470, 82.6879730, -110.0107422, 82.7921219, -192.6343689, 192.6987152
42: -79.7718201, 75.2875748, -79.9657288, 75.5157089, -155.2875366, 155.2532959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=562, inp2_unstable=563, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1206
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 349
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 316
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 365
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 284
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 285
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 286
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 351
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1125

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6476521, upper bound: 144.6253281
time: 249.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.6483345, upper bound: 144.8593562
time: 223.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -148.5568237, 80.7334137, -148.5701599, 80.7439423, -229.3007660, 229.3035583
1: -77.4305267, 74.4503098, -77.4376907, 74.4619217, -151.8924561, 151.8880005
2: -73.6089401, 66.0952606, -73.6153564, 66.1165009, -139.7254333, 139.7106018
3: -78.8872910, 83.0576324, -78.8932343, 83.0750580, -161.9623413, 161.9508667
4: -86.4402618, 82.2689285, -86.4505157, 82.2809753, -168.7212372, 168.7194519
5: -83.2953033, 87.2485962, -83.3016968, 87.2728119, -170.5681152, 170.5502777
6: -101.9223938, 87.9794464, -101.9345398, 87.9924545, -189.9148407, 189.9139709
7: -104.6600189, 88.9890442, -104.6691971, 89.0338593, -193.6938782, 193.6582336
8: -94.8271027, 101.9933167, -94.8370361, 102.0137482, -196.8408508, 196.8303375
9: -79.7468414, 82.3234558, -79.7663956, 82.3307877, -162.0776367, 162.0898438
10: -120.5159149, 113.7844772, -120.5408249, 113.7948303, -234.3107147, 234.3253021
11: -121.3257599, 90.6785736, -121.3381500, 90.6940460, -212.0197754, 212.0167236
12: -109.5420609, 115.0548706, -109.5641479, 115.0670700, -224.6091309, 224.6189880
13: -116.1772385, 122.5655212, -116.1922455, 122.5782547, -238.7554779, 238.7577667
14: -181.5201416, 112.2714081, -181.5424805, 112.2791672, -293.7992859, 293.8138733
15: -88.9151993, 81.1047211, -88.9538040, 81.1153717, -170.0305786, 170.0585327
16: -125.0478516, 92.4802246, -125.0649490, 92.5235443, -217.5713806, 217.5451660
17: -189.0287781, 152.8947754, -189.0489197, 152.9094849, -341.9382629, 341.9436646
18: -115.6695099, 89.5680389, -115.6831589, 89.5782318, -205.2477112, 205.2511902
19: -85.4019470, 41.5868797, -85.4121628, 41.6016998, -127.0036392, 126.9990387
20: -78.5199432, 56.5201111, -78.5296631, 56.5259895, -135.0459290, 135.0497589
21: -111.9892426, 60.3024902, -112.0014038, 60.3183212, -172.3075562, 172.3038940
22: -121.5577087, 73.7818146, -121.5815201, 73.7919006, -195.3496094, 195.3633423
23: -88.7239075, 59.0028915, -88.7336960, 59.0135117, -147.7374268, 147.7365875
24: -112.6423416, 63.3345566, -112.6535873, 63.3429794, -175.9853210, 175.9881287
25: -93.7574997, 66.2033234, -93.7816925, 66.2097931, -159.9672699, 159.9850159
26: -123.9597702, 103.4221420, -123.9824066, 103.4328003, -227.3925781, 227.4045410
27: -118.8211594, 83.6753845, -118.8332672, 83.6901550, -202.5113220, 202.5086365
28: -86.2319946, 66.4457550, -86.2399902, 66.4587860, -152.6907806, 152.6857452
29: -133.5018463, 82.0031586, -133.5201721, 82.0137558, -215.5155945, 215.5233002
30: -110.9609146, 85.6003952, -110.9713211, 85.6151352, -196.5760345, 196.5717163
31: -110.3712997, 56.2311745, -110.3839722, 56.2424355, -166.6137390, 166.6151428
32: -110.0174866, 87.4558868, -110.0321274, 87.4668121, -197.4842834, 197.4880066
33: -150.2037201, 109.0571899, -150.2201385, 109.0720520, -259.2757568, 259.2773438
34: -125.6753845, 92.9907455, -125.6864548, 93.0020142, -218.6773682, 218.6771851
35: -127.8876724, 91.1063080, -127.8996277, 91.1156845, -219.0033569, 219.0059052
36: -123.9735870, 98.7895203, -123.9899979, 98.7973175, -222.7708893, 222.7795105
37: -169.7675629, 97.7145996, -169.7843628, 97.7220001, -267.4895630, 267.4989624
38: -148.1941833, 120.8514481, -148.2071686, 120.8620300, -269.0562134, 269.0586243
39: -176.3788452, 113.7196808, -176.3953552, 113.7320786, -290.1109314, 290.1149902
40: -147.3050385, 95.2025146, -147.3200836, 95.2141342, -242.5191650, 242.5225983
41: -110.0945663, 82.8328857, -110.1075439, 82.8479004, -192.9424438, 192.9404297
42: -80.1132202, 75.5547791, -80.1248245, 75.5667191, -155.6799316, 155.6795959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=562, inp2_unstable=563, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1206
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 349
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 316
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 365
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 284
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 285
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 286
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 351
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1125

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6476521, upper bound: 144.6253281
time: 283.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.6483345, upper bound: 144.8647603
time: 228.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 514.14 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 514.14
Output dim: 12, lower bound: -144.6476521, upper bound: 144.6253281
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 514.14
Output dim: 12, lower bound: -144.6483345, upper bound: 144.8593562
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 514.14
Output dim: 12, lower bound: -144.6476521, upper bound: 144.6253281
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 514.14
Output dim: 12, lower bound: -144.6483345, upper bound: 144.8647603

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -148.3518066, 80.5216064, -148.4740448, 80.6429443, -228.9947205, 228.9956512
1: -77.3241119, 74.2686768, -77.3773117, 74.3819656, -151.7060699, 151.6459961
2: -73.4273987, 65.8776398, -73.5703964, 66.0083313, -139.4357300, 139.4480286
3: -78.6835861, 82.7705307, -78.8445892, 82.9182281, -161.6018066, 161.6151123
4: -86.1996841, 81.9007339, -86.3890533, 82.1014709, -168.3011475, 168.2897949
5: -83.1776657, 86.9982300, -83.2546692, 87.1492004, -170.3268585, 170.2528992
6: -101.6476440, 87.7111206, -101.7987061, 87.8862152, -189.5338440, 189.5098267
7: -104.4108963, 88.7214661, -104.5960007, 88.8917007, -193.3025818, 193.3174744
8: -94.5750046, 101.6623993, -94.7889099, 101.8563385, -196.4313354, 196.4513092
9: -79.6112137, 82.1287384, -79.7021179, 82.2402878, -161.8515015, 161.8308411
10: -120.1757355, 113.4233932, -120.3701248, 113.7004547, -233.8761902, 233.7935028
11: -121.0241089, 90.5208893, -121.1911469, 90.5904083, -211.6145020, 211.7120361
12: -108.6450806, 114.4439392, -109.1073227, 114.9958344, -223.6408997, 223.5512238
13: -115.7959747, 122.2191162, -115.9990158, 122.4880066, -238.2839813, 238.2181396
14: -180.7832794, 111.8303680, -181.1791077, 112.2391052, -293.0223999, 293.0094604
15: -88.6365433, 80.9635773, -88.8207397, 81.0295715, -169.6661072, 169.7843170
16: -124.7412109, 92.2229385, -124.9460449, 92.3806152, -217.1218262, 217.1689758
17: -188.1800232, 152.3583374, -188.6334381, 152.8522034, -341.0322266, 340.9917603
18: -115.3311615, 89.4209442, -115.5303345, 89.5237579, -204.8549194, 204.9512787
19: -85.2214508, 41.4924202, -85.3170776, 41.5491867, -126.7706299, 126.8094940
20: -78.2673645, 56.3778038, -78.4151917, 56.4792328, -134.7465973, 134.7929993
21: -111.7314911, 60.1834488, -111.8801498, 60.2651863, -171.9966736, 172.0635986
22: -121.0475464, 73.4383698, -121.3243942, 73.7282028, -194.7757568, 194.7627563
23: -88.4858856, 58.8858337, -88.6340027, 58.9512482, -147.4371185, 147.5198364
24: -112.3472443, 63.0902061, -112.5652618, 63.2196465, -175.5668945, 175.6554565
25: -93.5450439, 66.0662689, -93.6803360, 66.1501083, -159.6951599, 159.7465820
26: -123.2459183, 103.0974121, -123.6309052, 103.3764343, -226.6223450, 226.7283020
27: -118.5641708, 83.5226059, -118.7276306, 83.6092453, -202.1734161, 202.2502441
28: -86.0387726, 66.3272858, -86.1605835, 66.3952179, -152.4339905, 152.4878693
29: -132.9139252, 81.5641861, -133.2319183, 81.9651947, -214.8791199, 214.7961121
30: -110.7414093, 85.3251114, -110.8897705, 85.4847717, -196.2261810, 196.2148743
31: -110.0922012, 56.0085487, -110.2792511, 56.1256561, -166.2178650, 166.2877960
32: -109.6524734, 87.1595764, -109.8464813, 87.4118958, -197.0643463, 197.0060577
33: -149.7564545, 108.4822159, -150.1330719, 108.7750473, -258.5314941, 258.6152954
34: -125.3419647, 92.7083740, -125.6017380, 92.8528900, -218.1948547, 218.3101196
35: -127.5464020, 90.7847900, -127.8195801, 90.9407043, -218.4871063, 218.6043701
36: -123.6550980, 98.5414734, -123.8450012, 98.7447205, -222.3997955, 222.3864746
37: -169.3602295, 97.4866180, -169.6525269, 97.6065979, -266.9667969, 267.1390991
38: -147.8366852, 120.5819397, -148.0532379, 120.7675781, -268.6042480, 268.6351624
39: -176.0249634, 113.2653351, -176.2830353, 113.4928436, -289.5178223, 289.5483704
40: -146.9729462, 94.8378296, -147.2404633, 95.0214996, -241.9944458, 242.0782928
41: -109.8367996, 82.6811752, -109.9982758, 82.7767792, -192.6135864, 192.6794434
42: -79.7652283, 75.2744598, -79.9505920, 75.4850311, -155.2502441, 155.2250519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=562, inp2_unstable=562, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1206
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 349
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 399
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 300
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 317
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 316
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 365
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 268
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 278
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 267
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 284
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 285
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 286
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 351
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1125

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8513442
time: 2809.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8535485
time: 256.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -148.5476532, 80.7250977, -148.5493011, 80.7246475, -229.2723083, 229.2743835
1: -77.4250107, 74.4466705, -77.4251709, 74.4535065, -151.8785095, 151.8718262
2: -73.6040268, 66.0886002, -73.6041107, 66.1026154, -139.7066345, 139.6927032
3: -78.8817825, 83.0446320, -78.8808136, 83.0452576, -161.9270325, 161.9254456
4: -86.4291229, 82.2630005, -86.4248276, 82.2674332, -168.6965637, 168.6878357
5: -83.2899323, 87.2401505, -83.2896805, 87.2537994, -170.5437164, 170.5298157
6: -101.9156952, 87.9575882, -101.9192200, 87.9433746, -189.8590698, 189.8768005
7: -104.6522827, 88.9738922, -104.6515198, 88.9997253, -193.6520081, 193.6254120
8: -94.8199387, 101.9869843, -94.8207932, 101.9992981, -196.8192291, 196.8077393
9: -79.7394638, 82.3174667, -79.7495270, 82.3170166, -162.0564575, 162.0669861
10: -120.5041428, 113.7775879, -120.5135269, 113.7790451, -234.2831879, 234.2911072
11: -121.3155441, 90.6513443, -121.3145828, 90.6303101, -211.9458313, 211.9659271
12: -109.5284348, 115.0478668, -109.5325165, 115.0510941, -224.5795288, 224.5803528
13: -116.1665649, 122.5569839, -116.1681213, 122.5585556, -238.7251282, 238.7250977
14: -181.5035248, 112.2680817, -181.5050964, 112.2715073, -293.7750244, 293.7731934
15: -88.9028168, 81.0970764, -88.9260406, 81.0976715, -170.0004730, 170.0230865
16: -125.0384064, 92.4618225, -125.0433578, 92.4822006, -217.5205994, 217.5051575
17: -189.0159760, 152.8873291, -189.0197144, 152.8923340, -341.9082642, 341.9070435
18: -115.6601410, 89.5618134, -115.6614685, 89.5641937, -205.2243347, 205.2232666
19: -85.3965302, 41.5829086, -85.3998032, 41.5925598, -126.9890900, 126.9827118
20: -78.5142441, 56.5163422, -78.5165100, 56.5174446, -135.0316925, 135.0328522
21: -111.9811707, 60.2978745, -111.9831696, 60.3077126, -172.2888641, 172.2810364
22: -121.5407562, 73.7742538, -121.5420761, 73.7744904, -195.3152313, 195.3163300
23: -88.7185745, 58.9982567, -88.7214432, 59.0029869, -147.7215576, 147.7196960
24: -112.6346741, 63.3292885, -112.6361237, 63.3311386, -175.9658203, 175.9654083
25: -93.7474518, 66.1992416, -93.7587128, 66.2004852, -159.9479370, 159.9579468
26: -123.9462509, 103.4152603, -123.9509354, 103.4172058, -227.3634033, 227.3661804
27: -118.8130493, 83.6682892, -118.8147278, 83.6740494, -202.4870911, 202.4830017
28: -86.2276688, 66.4398346, -86.2301025, 66.4451752, -152.6728516, 152.6699371
29: -133.4918671, 81.9967804, -133.4971924, 81.9990692, -215.4909363, 215.4939728
30: -110.9529266, 85.5949249, -110.9530563, 85.6026840, -196.5556030, 196.5479736
31: -110.3636322, 56.2258377, -110.3664627, 56.2302017, -166.5938110, 166.5923004
32: -110.0104370, 87.4499969, -110.0159912, 87.4534454, -197.4638824, 197.4659576
33: -150.1957703, 109.0489197, -150.2020264, 109.0530014, -259.2487488, 259.2509155
34: -125.6688690, 92.9828644, -125.6716843, 92.9841156, -218.6529846, 218.6545410
35: -127.8805847, 91.0988235, -127.8835297, 91.0984955, -218.9790802, 218.9823608
36: -123.9677505, 98.7840347, -123.9766388, 98.7848816, -222.7526245, 222.7606812
37: -169.7585449, 97.7093735, -169.7637787, 97.7101593, -267.4686890, 267.4731445
38: -148.1868896, 120.8386230, -148.1904907, 120.8324509, -269.0193481, 269.0291138
39: -176.3574066, 113.7132187, -176.3457642, 113.7173615, -290.0747681, 290.0589600
40: -147.2970886, 95.1946106, -147.3018646, 95.1962357, -242.4933167, 242.4964752
41: -110.0891037, 82.8260803, -110.0950165, 82.8325653, -192.9216614, 192.9210968
42: -80.1065674, 75.5411987, -80.1096039, 75.5353241, -155.6418915, 155.6507874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=562, inp2_unstable=562, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1102
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1118
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1132
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1206
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 349
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1172
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 399
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 300
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 317
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 366
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 333
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 348
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 316
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 446
type: A, layer: 1, pos: 1101
type: A, layer: 1, pos: 332
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 350
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 365
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 417
type: A, layer: 1, pos: 268
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 278
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 301
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 267
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 284
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 270
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 266
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 445
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 285
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 286
type: A, layer: 1, pos: 274
type: A, layer: 1, pos: 351
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 433
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 1125

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.4207033, upper bound: 144.6069609
time: 459.04 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8590151
time: 276.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 737.99 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 737.99
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8513442
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 737.99
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8535485
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 737.99
Output dim: 12, lower bound: -144.4207033, upper bound: 144.6069609
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 737.99
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8590151

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -148.2539062, 80.3872833, -148.4239807, 80.5758820, -228.8297882, 228.8112488
1: -77.2643738, 74.1601868, -77.3477478, 74.3282776, -151.5926514, 151.5079346
2: -73.3801575, 65.6882095, -73.5470428, 65.9135208, -139.2936707, 139.2352448
3: -78.6346054, 82.5812531, -78.8206100, 82.8240128, -161.4586182, 161.4018402
4: -86.1484680, 81.7076416, -86.3630066, 82.0062561, -168.1547089, 168.0706482
5: -83.1278992, 86.7754517, -83.2294312, 87.0390472, -170.1669464, 170.0048828
6: -101.5613174, 87.6394119, -101.7556305, 87.8500671, -189.4113770, 189.3950500
7: -104.3369751, 88.4780273, -104.5597839, 88.7656097, -193.1025848, 193.0378113
8: -94.5293579, 101.4518051, -94.7661591, 101.7498856, -196.2792358, 196.2179413
9: -79.4619522, 82.0453339, -79.6280060, 82.1987610, -161.6607056, 161.6733398
10: -119.9115906, 113.3258743, -120.2395706, 113.6522980, -233.5638733, 233.5654449
11: -120.8880386, 90.4601593, -121.1237564, 90.5596924, -211.4477234, 211.5838928
12: -108.1003494, 114.3650894, -108.8397675, 114.9565887, -223.0568848, 223.2048645
13: -115.5556946, 122.1157684, -115.8798676, 122.4372101, -237.9928894, 237.9956360
14: -180.3843536, 111.7860641, -180.9820862, 112.2172623, -292.6016235, 292.7681580
15: -88.4312286, 80.8630219, -88.7170639, 80.9812393, -169.4124603, 169.5800781
16: -124.6048203, 92.0724487, -124.8794479, 92.3044357, -216.9092560, 216.9518738
17: -187.7814331, 152.2960205, -188.4369659, 152.8212891, -340.6027222, 340.7329712
18: -115.1611938, 89.3642273, -115.4449921, 89.4956131, -204.6567993, 204.8092194
19: -85.1175079, 41.4036789, -85.2644730, 41.5054932, -126.6230011, 126.6681519
20: -78.1590729, 56.3307152, -78.3611603, 56.4556694, -134.6147308, 134.6918793
21: -111.6049957, 60.0806427, -111.8163071, 60.2145271, -171.8195190, 171.8969421
22: -120.7875214, 73.3693542, -121.1922073, 73.6944580, -194.4819641, 194.5615540
23: -88.3814545, 58.7915573, -88.5819092, 58.9046707, -147.2861328, 147.3734436
24: -112.2616119, 62.9364090, -112.5225830, 63.1442223, -175.4058228, 175.4589844
25: -93.4573593, 66.0021133, -93.6350403, 66.1181488, -159.5755005, 159.6371460
26: -122.8074188, 103.0249481, -123.4125671, 103.3403244, -226.1477203, 226.4374847
27: -118.4555206, 83.3910980, -118.6733398, 83.5443268, -201.9998474, 202.0644379
28: -85.9466705, 66.2395782, -86.1144867, 66.3516235, -152.2982941, 152.3540649
29: -132.6839294, 81.5106201, -133.1140747, 81.9389725, -214.6228943, 214.6246948
30: -110.6575623, 85.1442719, -110.8482819, 85.3946075, -196.0521545, 195.9925537
31: -109.9733734, 55.8365364, -110.2209625, 56.0411377, -166.0145111, 166.0574646
32: -109.4527740, 87.1110687, -109.7481689, 87.3877563, -196.8405304, 196.8592377
33: -149.6661987, 108.2638092, -150.0884247, 108.6671524, -258.3333435, 258.3522339
34: -125.2388229, 92.6364594, -125.5509186, 92.8160248, -218.0548096, 218.1873474
35: -127.4624176, 90.6835785, -127.7780762, 90.8905487, -218.3529510, 218.4616394
36: -123.4854736, 98.4877930, -123.7598877, 98.7186584, -222.2041321, 222.2476807
37: -169.2249603, 97.4091187, -169.5855103, 97.5680237, -266.7929688, 266.9945984
38: -147.6832886, 120.5075531, -147.9769897, 120.7308884, -268.4141846, 268.4845581
39: -175.9375763, 113.1186905, -176.2395935, 113.4202118, -289.3577881, 289.3582764
40: -146.8937378, 94.6583862, -147.2010651, 94.9299164, -241.8236542, 241.8594513
41: -109.7299805, 82.6091385, -109.9451904, 82.7393188, -192.4692841, 192.5543213
42: -79.6004333, 75.2070312, -79.8694077, 75.4517899, -155.0522156, 155.0764160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=561, inp2_unstable=562, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=832, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1206
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 349
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 316
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 365
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 284
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 285
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 286
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 351
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1125

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 649

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3661454
time: 270.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.4011183, upper bound: 144.6159354
time: 205.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -148.4520569, 80.5449677, -148.4632568, 80.6310425, -229.0830994, 229.0082245
1: -77.3791504, 74.2833557, -77.3707886, 74.3660202, -151.7451782, 151.6541138
2: -73.5864563, 65.8946381, -73.5654221, 66.0006027, -139.5870514, 139.4600525
3: -78.8405914, 82.7937393, -78.8397675, 82.9062271, -161.7468109, 161.6335144
4: -86.3153381, 81.9285126, -86.3839874, 82.0919189, -168.4072571, 168.3125000
5: -83.3320389, 87.0249634, -83.2502289, 87.1401367, -170.4721680, 170.2751923
6: -101.6880722, 87.7315826, -101.7912903, 87.8701401, -189.5581970, 189.5228729
7: -104.6780701, 88.7324982, -104.5876465, 88.8713379, -193.5494080, 193.3201294
8: -94.7419739, 101.6950150, -94.7824554, 101.8467789, -196.5887451, 196.4774780
9: -79.6410065, 82.2886353, -79.6953964, 82.2333298, -161.8743134, 161.9840393
10: -120.2174377, 113.6907959, -120.3584900, 113.6929321, -233.9103394, 234.0492859
11: -121.0333710, 90.4831924, -121.1801147, 90.5460281, -211.5793915, 211.6632843
12: -108.6695709, 114.9396591, -109.0895309, 114.9884491, -223.6580200, 224.0291901
13: -115.8286285, 122.4864197, -115.9882584, 122.4775848, -238.3061829, 238.4746704
14: -180.8515930, 112.1716919, -181.1617432, 112.2349319, -293.0865173, 293.3334351
15: -88.6613388, 81.1036224, -88.7940140, 81.0203552, -169.6816864, 169.8976288
16: -124.8737488, 92.2096481, -124.9357452, 92.3471603, -217.2208862, 217.1453857
17: -188.2268066, 152.7080078, -188.6183777, 152.8436279, -341.0704041, 341.3263855
18: -115.4337006, 89.4574432, -115.5203552, 89.5178680, -204.9515533, 204.9777985
19: -85.2794800, 41.4967651, -85.3107758, 41.5446701, -126.8241501, 126.8075409
20: -78.3327255, 56.4053993, -78.4087906, 56.4753914, -134.8081055, 134.8141937
21: -111.7830658, 60.2058029, -111.8712158, 60.2605591, -172.0436249, 172.0770264
22: -121.0579376, 73.6437683, -121.2942047, 73.7192917, -194.7772217, 194.9379578
23: -88.5670166, 58.8944664, -88.6278915, 58.9464149, -147.5134277, 147.5223541
24: -112.5224380, 63.1090927, -112.5558624, 63.2141190, -175.7365570, 175.6649475
25: -93.5690079, 66.1052094, -93.6668777, 66.1457672, -159.7147827, 159.7720795
26: -123.3104401, 103.4017334, -123.6136932, 103.3682251, -226.6786499, 227.0154266
27: -118.7367859, 83.5340881, -118.7175446, 83.6020889, -202.3388672, 202.2516174
28: -86.1147537, 66.3428802, -86.1557617, 66.3899612, -152.5047150, 152.4986267
29: -132.9532776, 81.8500366, -133.2180176, 81.9577484, -214.9110260, 215.0680389
30: -110.8437119, 85.3530731, -110.8804169, 85.4756927, -196.3193970, 196.2334900
31: -110.2742462, 56.0220604, -110.2699203, 56.1184845, -166.3927155, 166.2919769
32: -109.6851501, 87.3249207, -109.8366470, 87.4062729, -197.0914307, 197.1615601
33: -149.9688721, 108.5097809, -150.1249695, 108.7654572, -258.7343140, 258.6347351
34: -125.4614944, 92.7296600, -125.5941925, 92.8468781, -218.3083801, 218.3238525
35: -127.6737442, 90.7958679, -127.8120346, 90.9351883, -218.6089172, 218.6078796
36: -123.6877823, 98.6899261, -123.8275452, 98.7392349, -222.4270172, 222.5174713
37: -169.4931641, 97.5117035, -169.6426239, 97.6019287, -267.0950928, 267.1543274
38: -147.9038239, 120.6908569, -148.0373840, 120.7617569, -268.6655884, 268.7282104
39: -176.1147308, 113.2915802, -176.2748260, 113.4863510, -289.6010742, 289.5664062
40: -147.1586609, 94.8485260, -147.2310181, 95.0140686, -242.1727295, 242.0795288
41: -109.9027557, 82.7015533, -109.9909821, 82.7672729, -192.6699982, 192.6925049
42: -79.7950592, 75.3803482, -79.9427490, 75.4792633, -155.2743073, 155.3230896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=561, inp2_unstable=562, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1102
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1118
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 1132
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1206
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 349
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1172
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 399
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 300
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 317
type: B, layer: 1, pos: 366
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 333
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 348
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 316
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 446
type: B, layer: 1, pos: 1101
type: B, layer: 1, pos: 332
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 350
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 365
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 417
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 278
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 301
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 284
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 270
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 266
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 445
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 285
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 286
type: B, layer: 1, pos: 274
type: B, layer: 1, pos: 351
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 433
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 1125

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 649

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3661454
time: 4678.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3739592
time: 165.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4846.34 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4846.34
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3661454
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4846.34
Output dim: 12, lower bound: -144.4011183, upper bound: 144.6159354
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4846.34
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3661454
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4846.34
Output dim: 12, lower bound: -144.4011183, upper bound: 144.3739592
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4846.34
Output dim: 12, lower bound: -144.4237686, upper bound: 144.8590151

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 454.52 + 10567.35 = 11021.87 seconds
