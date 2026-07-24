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
execution time: IAR + RelationalAnalysis = 2.97 + 463.70 = 466.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -144.8773035, upper bound: 144.8773035

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8768012, upper bound: 144.6208291
time: 761.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.6208291, upper bound: 144.8768012
time: 287.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1048.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1048.52
Output dim: 12, lower bound: -144.8768012, upper bound: 144.6208291
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1048.52
Output dim: 12, lower bound: -144.6208291, upper bound: 144.8768012

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6650634, upper bound: 144.6108092
time: 194.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8683677, upper bound: 144.4020213
time: 421.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.4020213, upper bound: 144.8683677
time: 1888.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6108092, upper bound: 144.6650634
time: 455.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2346.92 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2346.92
Output dim: 12, lower bound: -144.6650634, upper bound: 144.6108092
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2346.92
Output dim: 12, lower bound: -144.8683677, upper bound: 144.4020213
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2346.92
Output dim: 12, lower bound: -144.4020213, upper bound: 144.8683677
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2346.92
Output dim: 12, lower bound: -144.6108092, upper bound: 144.6650634

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6357771, upper bound: 144.3717813
time: 196.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8642657, upper bound: 144.3044295
time: 890.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 663

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.3044295, upper bound: 144.8642657
time: 190.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.3717813, upper bound: 144.6357771
time: 365.19 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 557.96 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 557.96
Output dim: 12, lower bound: -144.6357771, upper bound: 144.3717813
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 557.96
Output dim: 12, lower bound: -144.8642657, upper bound: 144.3044295
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 557.96
Output dim: 12, lower bound: -144.3044295, upper bound: 144.8642657
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 557.96
Output dim: 12, lower bound: -144.3717813, upper bound: 144.6357771

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8587417, upper bound: 144.2294394
time: 264.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.6707983, upper bound: 144.2971005
time: 235.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.2971005, upper bound: 144.6707983
time: 226.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.2294394, upper bound: 144.8587417
time: 162.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 391.88 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 391.88
Output dim: 12, lower bound: -144.8587417, upper bound: 144.2294394
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 391.88
Output dim: 12, lower bound: -144.6707983, upper bound: 144.2971005
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 391.88
Output dim: 12, lower bound: -144.2971005, upper bound: 144.6707983
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 391.88
Output dim: 12, lower bound: -144.2294394, upper bound: 144.8587417

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -148.5830231, 80.7540741, -148.5830231, 80.7540741, -229.3370972, 229.3370972
1: -77.4445801, 74.4728470, -77.4445801, 74.4728470, -151.9173889, 151.9174194
2: -73.6215973, 66.1374817, -73.6215973, 66.1374817, -139.7590790, 139.7590790
3: -78.8991699, 83.0912170, -78.8991699, 83.0912170, -161.9903870, 161.9903870
4: -86.4604034, 82.2945862, -86.4604034, 82.2945862, -168.7549744, 168.7549896
5: -83.3080750, 87.2951202, -83.3080750, 87.2951202, -170.6031952, 170.6031952
6: -101.9465103, 88.0049896, -101.9465103, 88.0049896, -189.9515076, 189.9515076
7: -104.6780853, 89.0748520, -104.6780853, 89.0748520, -193.7529297, 193.7529297
8: -94.8466797, 102.0342560, -94.8466797, 102.0342560, -196.8809357, 196.8809204
9: -79.7845917, 82.3380508, -79.7845917, 82.3380508, -162.1226501, 162.1226501
10: -120.5665741, 113.8047943, -120.5665741, 113.8047943, -234.3713684, 234.3713379
11: -121.3502045, 90.7091370, -121.3502045, 90.7091370, -212.0593109, 212.0593262
12: -109.5845795, 115.0789566, -109.5845795, 115.0789566, -224.6635437, 224.6635437
13: -116.2064209, 122.5904541, -116.2064209, 122.5904541, -238.7968750, 238.7968750
14: -181.5636902, 112.2865906, -181.5636902, 112.2865906, -293.8502808, 293.8502808
15: -88.9907837, 81.1258011, -88.9907837, 81.1258011, -170.1165771, 170.1165771
16: -125.0813446, 92.5634384, -125.0813446, 92.5634384, -217.6447754, 217.6447754
17: -189.0679169, 152.9237061, -189.0679169, 152.9237061, -341.9916077, 341.9915771
18: -115.6960983, 89.5881119, -115.6960983, 89.5881119, -205.2842102, 205.2842102
19: -85.4221878, 41.6164856, -85.4221878, 41.6164856, -127.0386734, 127.0386734
20: -78.5391159, 56.5315323, -78.5391159, 56.5315323, -135.0706482, 135.0706482
21: -112.0131836, 60.3335114, -112.0131836, 60.3335114, -172.3466949, 172.3466949
22: -121.6092148, 73.8017044, -121.6092148, 73.8017044, -195.4109039, 195.4109192
23: -88.7431030, 59.0246239, -88.7431030, 59.0246239, -147.7677307, 147.7677307
24: -112.6645050, 63.3509598, -112.6645050, 63.3509598, -176.0154572, 176.0154724
25: -93.8043060, 66.2160950, -93.8043060, 66.2160950, -160.0204010, 160.0204010
26: -124.0044327, 103.4432678, -124.0044327, 103.4432678, -227.4476929, 227.4476929
27: -118.8448868, 83.7040100, -118.8448868, 83.7040100, -202.5488739, 202.5488892
28: -86.2476196, 66.4726257, -86.2476196, 66.4726257, -152.7202454, 152.7202454
29: -133.5373383, 82.0239105, -133.5373383, 82.0239105, -215.5612183, 215.5612488
30: -110.9814148, 85.6290970, -110.9814148, 85.6290970, -196.6105042, 196.6105042
31: -110.3965302, 56.2552185, -110.3965302, 56.2552185, -166.6517487, 166.6517487
32: -110.0461121, 87.4772949, -110.0461121, 87.4772949, -197.5234070, 197.5234070
33: -150.2357025, 109.0859680, -150.2357025, 109.0859680, -259.3216553, 259.3216553
34: -125.6970520, 93.0128479, -125.6970520, 93.0128479, -218.7098694, 218.7098694
35: -127.9108734, 91.1245270, -127.9108734, 91.1245270, -219.0353851, 219.0354004
36: -124.0052490, 98.8048630, -124.0052490, 98.8048630, -222.8101196, 222.8101196
37: -169.8002014, 97.7293091, -169.8002014, 97.7293091, -267.5295105, 267.5295105
38: -148.2205658, 120.8721695, -148.2205658, 120.8721695, -269.0927429, 269.0927124
39: -176.4112244, 113.7436829, -176.4112244, 113.7436829, -290.1548767, 290.1549072
40: -147.3345947, 95.2253265, -147.3345947, 95.2253265, -242.5599213, 242.5599060
41: -110.1204834, 82.8623657, -110.1204834, 82.8623657, -192.9828491, 192.9828491
42: -80.1361542, 75.5783386, -80.1361542, 75.5783386, -155.7144928, 155.7144928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=563, inp2_unstable=563, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=833, inp2_unstable=833, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1206
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 433
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1172
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1102
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 403
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 382
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 383
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 239
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 191
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 318
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1132
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 398
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 348
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 285
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 463
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 349
type: RSZ, layer: 1, pos: 339
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 365
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1117
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 335
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 286
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 446
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 351
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 462
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 334
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 417
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 332
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 350
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 284
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 495
type: RSZ, layer: 1, pos: 316
type: RSZ, layer: 1, pos: 278
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 402
type: RSZ, layer: 1, pos: 302
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 317
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 385
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 270
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1101
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 386
type: RSZ, layer: 1, pos: 445
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 366
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 268
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 266
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 267
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 355
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 367
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 418
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 498
type: RSZ, layer: 1, pos: 369
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1125
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 511
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 300
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 274
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 333
type: RSZ, layer: 1, pos: 301
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 1636

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -144.5852791, upper bound: 144.2168375
time: 230.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -144.8529597, upper bound: 144.0515617
time: 500.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 732.98 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 732.98
Output dim: 12, lower bound: -144.5852791, upper bound: 144.2168375
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 732.98
Output dim: 12, lower bound: -144.8529597, upper bound: 144.0515617
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 732.98
Output dim: 12, lower bound: -144.2294394, upper bound: 144.8587417

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 466.67 + 7289.19 = 7755.86 seconds
